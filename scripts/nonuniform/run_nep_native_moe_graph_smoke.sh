#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=2
#SBATCH --segment=1
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.native-moe-graph
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
export CODE_REPO="${GRAPH_CODE_REPO:-$base/.worktrees/nep-slot-cache-20260917}"
export GRAPH_FULL_EP=${GRAPH_FULL_EP:-4} GRAPH_REDUCED_EP=${GRAPH_REDUCED_EP:-2}
export CUDA_DEVICE_MAX_CONNECTIONS=32 NCCL_LAUNCH_ORDER_IMPLICIT=1 OMP_NUM_THREADS=1
export NCCL_NVLS_ENABLE=0 TORCH_NCCL_BLOCKING_WAIT=0 USE_MNNVL=1
export NVTE_FUSED_ATTN=0 TORCHINDUCTOR_WORKER_START=fork
export PROFILE_ACTIVITY_MODE=cupti CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0
export CUDA_EVENT_POOL_SIZE=16384
export PYTHONPATH="$CODE_REPO:$base/scripts/nonuniform:${PYTHONPATH:-}"

if [[ "${NATIVE_GRAPH_RANK:-0}" == 1 ]]; then
    export RANK=$SLURM_PROCID LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    # Slurm's task environment may replace WORLD_SIZE with the step task count.
    export WORLD_SIZE=$((2 * GRAPH_FULL_EP)) MASTER_ADDR=$GRAPH_MASTER_ADDR MASTER_PORT=$GRAPH_MASTER_PORT
    [[ "$GRAPH_CASE" != nep ]] || export WORLD_SIZE=$((GRAPH_FULL_EP + GRAPH_REDUCED_EP))
    # Keep full-replica placement fixed; unused trailing tasks exit for NEP.
    ((RANK < WORLD_SIZE)) || exit 0
    replica=$((RANK / GRAPH_FULL_EP))
    export NCCL_MNNVL_CLIQUE_ID=$((1000 + replica))
    export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$GRAPH_FULL_EP
    if [[ "$GRAPH_CASE" == nep && "$replica" == 1 ]]; then
        export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$GRAPH_REDUCED_EP
    fi
    export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,ENV
    export NCCL_DEBUG_FILE="$GRAPH_OUTPUT/transport/nccl.%h.%p.log"
    export CUPTI_OUTPUT_DIR="$GRAPH_OUTPUT/cupti_profile"
    printf 'rank=%s host=%s gpu=%s replica=%s clique=%s\n' "$RANK" "$(hostname -s)" "$CUDA_VISIBLE_DEVICES" "$replica" "$NCCL_MNNVL_CLIQUE_ID" > "$GRAPH_OUTPUT/placement/rank-$RANK.txt"
    gbs=$((WORLD_SIZE / 2 * 2 * 2))
    args=(
        --use-mcore-models --hybrid-layer-pattern E
        --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
        --hidden-size 4608 --ffn-hidden-size 3072 --moe-latent-size 1152
        --num-attention-heads 40 --kv-channels 128 --group-query-attention --num-query-groups 8
        --mamba-num-heads 128 --attention-backend flash --squared-relu --use-fused-weighted-squared-relu
        --normalization RMSNorm --disable-bias-linear --position-embedding-type none
        --untie-embeddings-and-output-weights --attention-dropout 0 --hidden-dropout 0
        --num-experts 512 --moe-router-topk 6 --moe-shared-expert-intermediate-size 6144
        --moe-grouped-gemm --moe-router-dtype fp32 --moe-router-score-function sigmoid
        --moe-router-force-uniform-routing --moe-router-load-balancing-type none
        --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep
        --moe-permute-fusion
        --bf16 --tensor-model-parallel-size 2 --context-parallel-size 1
        --expert-model-parallel-size "$GRAPH_FULL_EP" --expert-tensor-parallel-size 1 --sequence-parallel
        --pipeline-model-parallel-size 1 --micro-batch-size 2 --global-batch-size "$gbs"
        --seq-length 1024 --max-position-embeddings 1024 --train-iters 12
        --mock-data --tokenizer-type NullTokenizer --vocab-size 4096 --num-workers 1
        --no-create-attention-mask-in-dataloader --eval-iters 0 --eval-interval 1000
        --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather --ddp-num-buckets 2
        --cuda-graph-impl local --cuda-graph-modules "${GRAPH_SCOPE/dropless/moe_router}" --cuda-graph-warmup-steps 3
        --recompute-granularity selective --recompute-modules moe --te-rng-tracker
        --lr 0.0001 --lr-decay-iters 12 --lr-warmup-iters 1 --seed 1234
        --calculate-per-token-loss --no-check-for-nan-in-loss-and-grad --rerun-mode disabled
        --nonuniform-disable-nongrad-sync-collectives --distributed-timeout-minutes 3
        --profile --use-pytorch-profiler --profile-step-start 8 --profile-step-end 11
        --profile-ranks $(seq -s ' ' 0 $((WORLD_SIZE - 1)))
        --log-interval 1 --log-memory-interval 1 --tensorboard-dir "$GRAPH_OUTPUT/tensorboard"
    )
    if [[ "$GRAPH_SCOPE" != dropless ]]; then
        args+=(--moe-expert-capacity-factor 1 --moe-pad-expert-input-to-capacity)
    fi
    if [[ "$GRAPH_CASE" == nep ]]; then
        args+=(--nonuniform-mode ep --nonuniform-ep-ddp-approach nccl
               --nonuniform-ep-num-tp-cp-per-replica "$((GRAPH_FULL_EP / 2))" "$((GRAPH_REDUCED_EP / 2))")
    fi
    printf '%q ' "${args[@]}" > "$GRAPH_OUTPUT/args-rank-$RANK.txt"
    cd "$CODE_REPO"
    exec python -u -c 'from pretrain_a8b_remaining_overheads import install_scopes; install_scopes(); from pretrain_profile_activity_probe import main; main()' "${args[@]}"
fi

root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/native_moe_graph/$SLURM_JOB_ID
mkdir -p "$root"
if [[ "${GRAPH_ALLOW_DIRTY:-0}" != 1 ]]; then
    git -C "$CODE_REPO" diff --quiet HEAD
fi
git -C "$CODE_REPO" rev-parse HEAD > "$root/commit.txt"
scontrol show job "$SLURM_JOB_ID" > "$root/allocation.txt"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export GRAPH_MASTER_ADDR=${nodes[0]}
container=(--container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06
           --no-container-mount-home --container-mounts="$base:$base,/lustre:/lustre")
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --mpi=none "${container[@]}" \
    bash -lc "cd '$CODE_REPO'; make -C megatron/core/datasets; python -c 'import torch, transformer_engine; print(torch.__version__)'"
if [[ "${GRAPH_UNIT_TESTS:-0}" == 1 ]]; then
    srun --nodes=1 --ntasks=1 --mpi=none "${container[@]}" bash -lc "
        set -e
        cd '$CODE_REPO'
        uv run --no-project --active isort tests/unit_tests/distributed/test_nonuniform_ep.py
        python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py -k 'hybridep_static_capacity or flex_postprocess_cached_slot_indices or flex_metadata_maps_logical_experts_to_padded_ep6_slots'
    " > "$root/unit_tests.log" 2>&1
fi
git -C "$CODE_REPO" diff HEAD > "$root/implementation.patch"
sha256sum "$CODE_REPO/megatron/core/transformer/moe/token_dispatcher.py" > "$root/dispatcher.sha256"
export NATIVE_GRAPH_RANK=1
index=0
for GRAPH_CASE in ${GRAPH_CASES:-healthy nep}; do
    export GRAPH_CASE
    for GRAPH_SCOPE in ${GRAPH_SCOPES:-moe_router moe}; do
        export GRAPH_SCOPE GRAPH_OUTPUT="$root/${GRAPH_CASE}_${GRAPH_SCOPE}"
        export GRAPH_MASTER_PORT=$((32210 + index))
        index=$((index + 1))
        mkdir -p "$GRAPH_OUTPUT"/{transport,placement,cupti_profile}
        echo "NATIVE_GRAPH_BEGIN $GRAPH_CASE $GRAPH_SCOPE $(date --iso-8601=seconds)"
        timeout --signal=TERM --kill-after=30s 5m srun --nodes="$SLURM_NNODES" --ntasks="$((2 * GRAPH_FULL_EP))" --ntasks-per-node=4 \
            --kill-on-bad-exit=1 --mpi=none "${container[@]}" \
            bash "$base/scripts/nonuniform/run_nep_native_moe_graph_smoke.sh" \
            > "$GRAPH_OUTPUT/driver.log" 2>&1
        echo "NATIVE_GRAPH_DONE $GRAPH_CASE $GRAPH_SCOPE $(date --iso-8601=seconds)"
    done
done
