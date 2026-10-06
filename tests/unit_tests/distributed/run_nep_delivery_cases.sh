#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=2
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.cleanup-ab
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
runner=$base/.worktrees/nep-norm-headroom-20261006/tests/unit_tests/distributed/run_nep_delivery_cases.sh
export CODE_REPO="${CODE_REPO:-${GRAPH_CODE_REPO:-$base/.worktrees/nep-buffer-cleanup-20260921}}"
export GRAPH_FULL_EP=${GRAPH_FULL_EP:-8} GRAPH_REDUCED_EP=${GRAPH_REDUCED_EP:-6}
export CUDA_DEVICE_MAX_CONNECTIONS=32 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SLURM_CPU_BIND=none
# The prior EP8/6 smoke required this to avoid native dense-RS registration error 524.
# Do not propagate this small-test override to the original EP16/12 launcher.
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-1}
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0
export MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0 MEGATRON_NONUNIFORM_EP_BUCKET_READY_GATHER=1
export MEGATRON_NONUNIFORM_EP_DEVICE_ORDERED_EDP=1 MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE=0
export MEGATRON_NONUNIFORM_EP_SAME_COMM_READY=0 MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH=0
export MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE=1 MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER=0
export MEGATRON_NONUNIFORM_EP_END_ITERATION_SCATTER=1 MEGATRON_NONUNIFORM_EP_PARALLEL_GATHER_WINDOW=1
export MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0 MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES=1
export MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0 MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES=0
export MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS=1 MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=64 MEGATRON_NONUNIFORM_EP_NCCL_GATHER_BUCKETS_PER_EDP=1
export MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES=8589934592 MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER=0
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK=1 MEGATRON_NONUNIFORM_EP_BENCHMARK_PHASE_LIMIT=scatter
export NCCL_NVLS_ENABLE=0 TORCH_NCCL_BLOCKING_WAIT=0 USE_MNNVL=1
export NVTE_FUSED_ATTN=0 TORCHINDUCTOR_WORKER_START=fork
export PROFILE_ACTIVITY_MODE=cupti CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0
export CUDA_EVENT_POOL_SIZE=${CUDA_EVENT_POOL_SIZE:-16384}
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
    mbs=${ABLATION_MBS:-2}
    seqlen=${ABLATION_SEQLEN:-1024}
    ga=${ABLATION_GA:-2}
    if [[ "$GRAPH_CASE" == nep && "$replica" == 1 ]]; then
        ga=${ABLATION_REDUCED_GA:-$ga}
    fi
    gbs=$((WORLD_SIZE / 2 * mbs * ga))
    echo "BATCH_SCHEDULE rank=$RANK replica=$replica mbs=$mbs ga=$ga calculator_gbs=$gbs true_gbs=${TRUE_GLOBAL_BATCH_SIZE:-$gbs}"
    args=(
        --use-mcore-models --hybrid-layer-pattern "${ABLATION_LAYER_PATTERN:-EEEE}"
        --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
        --hidden-size 4608 --ffn-hidden-size 3072 --moe-latent-size 1152
        --num-attention-heads 40 --kv-channels 128 --group-query-attention --num-query-groups 8
        --mamba-num-heads 128 --attention-backend flash --squared-relu --use-fused-weighted-squared-relu
        --normalization RMSNorm --disable-bias-linear --position-embedding-type none
        --untie-embeddings-and-output-weights --attention-dropout 0 --hidden-dropout 0
        --num-experts 512 --moe-router-topk 6 --moe-shared-expert-intermediate-size 6144
        --moe-grouped-gemm --moe-router-dtype fp32 --moe-router-score-function sigmoid
        --moe-router-load-balancing-type none
        --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep
        --moe-permute-fusion
        --bf16 --tensor-model-parallel-size 2 --context-parallel-size 1
        --expert-model-parallel-size "$GRAPH_FULL_EP" --expert-tensor-parallel-size 1 --sequence-parallel
        --pipeline-model-parallel-size 1 --micro-batch-size "$mbs" --global-batch-size "$gbs"
        --seq-length "$seqlen" --max-position-embeddings "$seqlen" --train-iters 25
        --mock-data --tokenizer-type NullTokenizer --vocab-size 4096 --num-workers 1
        --no-create-attention-mask-in-dataloader --eval-iters 0 --eval-interval 1000
        --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather --ddp-num-buckets 2
        --cuda-graph-impl local --cuda-graph-modules "${GRAPH_SCOPE/dropless/moe_router}" --cuda-graph-warmup-steps 3
        --recompute-granularity selective --recompute-modules moe --te-rng-tracker
        --lr 0.0001 --lr-decay-iters 25 --lr-warmup-iters 1 --seed 1234
        --calculate-per-token-loss --no-check-for-nan-in-loss-and-grad --rerun-mode disabled
        --nonuniform-disable-nongrad-sync-collectives --distributed-timeout-minutes 3
        --profile --use-pytorch-profiler --profile-step-start 15 --profile-step-end 21
        --profile-ranks $(seq -s ' ' 0 $((WORLD_SIZE - 1)))
        --log-interval 1 --log-memory-interval 1 --tensorboard-dir "$GRAPH_OUTPUT/tensorboard"
    )
    if [[ "$GRAPH_SCOPE" != dropless ]]; then
        args+=(--moe-expert-capacity-factor 1 --moe-pad-expert-input-to-capacity)
    fi
    if [[ "${ABLATION_ROUTING:-uniform}" == uniform ]]; then
        args+=(--moe-router-force-uniform-routing)
    else
        [[ "$ABLATION_ROUTING" == random && "$GRAPH_SCOPE" == dropless ]]
    fi
    if [[ "$GRAPH_CASE" == nep ]]; then
        args+=(--nonuniform-mode ep --nonuniform-ep-ddp-approach nccl
               --nonuniform-ep-num-tp-cp-per-replica "$((GRAPH_FULL_EP / 2))" "$((GRAPH_REDUCED_EP / 2))")
    fi
    if [[ "${NEP_ACTUAL_A8B:-0}" == 1 ]]; then
        python3 "$CODE_REPO/tests/unit_tests/distributed/nep_actual_ablation.py" options > "$GRAPH_OUTPUT/options-$RANK.nul"
        mapfile -d '' -t args < "$GRAPH_OUTPUT/options-$RANK.nul"
    fi
    printf '%q ' "${args[@]}" > "$GRAPH_OUTPUT/args-rank-$RANK.txt"
    env | sort > "$GRAPH_OUTPUT/env-rank-$RANK.txt"
    cd "$CODE_REPO"
    export PROFILE_ACTIVITY_MODE=nsys NSYS_OUTPUT_DIR="$GRAPH_OUTPUT/nsys_profile"
    export NSYS_TMPDIR="$NSYS_OUTPUT_DIR/tmp-rank-$RANK"
    mkdir -p "$NSYS_TMPDIR"
    probe=${CONTROLLED_ENTRYPOINT:-$base/scripts/nonuniform/pretrain_nep_cleanup_probe.py}
    if [[ "${NEP_CONTROLLED_PROFILE:-on}" == off ]]; then
        python -u "$probe" "${args[@]}"
        exit
    fi
    # Device-side event tracing can introduce false stream dependencies and
    # hung the MBS1/GA16 graph run. Custom AG/RS completion events remain enabled.
    TMPDIR="$NSYS_TMPDIR" nsys profile --trace=cuda-sw,nvtx --nvtx-domain-include=nep_capture \
        --sample=none --cpuctxsw=none --cuda-graph-trace=node --cuda-event-trace="${NSYS_CUDA_EVENT_TRACE:-false}" \
        --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
        --output="$NSYS_OUTPUT_DIR/rank-$RANK" \
        env TMPDIR=/tmp python -u "$probe" "${args[@]}"
    nsys export --type=sqlite --force-overwrite=true --output="$NSYS_OUTPUT_DIR/rank-$RANK.sqlite" \
        "$NSYS_OUTPUT_DIR/rank-$RANK.nsys-rep"
    exit
fi

root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/$SLURM_JOB_ID
mkdir -p "$root"
if [[ "${GRAPH_ALLOW_DIRTY:-1}" != 1 ]]; then
    git -C "$CODE_REPO" diff --quiet HEAD
fi
git -C "$CODE_REPO" rev-parse HEAD > "$root/commit.txt"
scontrol show job "$SLURM_JOB_ID" > "$root/allocation.txt"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
python3 "$base/scripts/nonuniform/nep_nvl_placement.py" --prepare "$root/domains.json"
python3 - "$root/domains.json" "${nodes[@]}" <<'PY'
import json, sys
domains = json.load(open(sys.argv[1]))
hosts = sys.argv[2:]
half = len(hosts) // 2
for replica, members in enumerate((hosts[:half], hosts[half:])):
    blocks = {domains[host] for host in members}
    assert len(blocks) == 1, (replica, members, blocks)
    print('REPLICA_PLACEMENT', replica, members, blocks, flush=True)
PY
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
export NCCL_LAUNCH_ORDER_IMPLICIT=0
srun --nodes=1 --ntasks=1 --mpi=none "${container[@]}" bash -lc "
    set -e
    cd '$CODE_REPO'
    export CUPTI_OUTPUT_DIR='$root/compact_correctness'
    timeout --signal=TERM --kill-after=10s 180 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/unit_tests/distributed/nep_compact_params_smoke.py
    for policy in early consumer noprefetch_early noprefetch_consumer; do
        export NEP_ABLATION_PARAM_RELEASE=early NEP_ABLATION_PREFETCH_DEPTH=1 NEP_ABLATION_PARAM_LANES=1
        if [[ \$policy == *consumer ]]; then export NEP_ABLATION_PARAM_RELEASE=consumer; fi
        if [[ \$policy == noprefetch* ]]; then export NEP_ABLATION_PREFETCH_DEPTH=0; fi
        export CUPTI_OUTPUT_DIR='$root/prefetch_correctness'-\$policy
        timeout --signal=TERM --kill-after=10s 180 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/unit_tests/distributed/nep_param_prefetch_smoke.py
    done
" > "$root/correctness.log" 2>&1
export NATIVE_GRAPH_RANK=1
index=0
for implementation in ${CLEANUP_CASES:-reference cleanup}; do
    export NVTE_BWD_LAYERNORM_SM_MARGIN=0
    case "$implementation" in
        controlled_*)
            export CODE_REPO="${CONTROLLED_CODE_REPO:-$base/.worktrees/nep-placement-controlled-20260929}" NVTE_BWD_LAYERNORM_SM_MARGIN=32
            export CONTROLLED_ENTRYPOINT="$CODE_REPO/tests/unit_tests/distributed/nep_controlled_probe.py"
            export NEP_ABLATION_PLACEMENT=compact NEP_ABLATION_PARAM_ORDER=waves NEP_ABLATION_PARAM_LANES=1
            export NEP_ABLATION_FIXED_GRAD_ORDER=0 NEP_CONTROLLED_PROFILE=on GRAPH_CASES=nep
            export NEP_ABLATION_PARAM_RELEASE=early NEP_ABLATION_PREFETCH_DEPTH=1
            export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=3
            policy=${implementation#controlled_}
            if [[ "$policy" == *_off ]]; then
                export NEP_CONTROLLED_PROFILE=off
                policy=${policy%_off}
            fi
            case "$policy" in
                delivery_*)
                    export NEP_CONTROLLED_ROUTING_SEED=101 NEP_ABLATION_FIXED_GRAD_ORDER=1
                    export NEP_CONTROLLED_REPLAY=0 NEP_CONTROLLED_ROUTE_UNIT=512
                    [[ "$policy" != delivery_consumer ]] || export NEP_ABLATION_PARAM_RELEASE=consumer
                    [[ "$policy" != delivery_deep ]] || export NEP_ABLATION_PREFETCH_DEPTH=2
                    [[ "$policy" != *noprefetch* ]] || export NEP_ABLATION_PREFETCH_DEPTH=0
                    [[ "$policy" != *consumer ]] || export NEP_ABLATION_PARAM_RELEASE=consumer
                    [[ "$policy" != delivery_healthy* ]] || export GRAPH_CASES=healthy
                    ;;
                actual_*)
                    IFS=_ read -r _ cell repeat <<< "$policy"
                    [[ "$repeat" =~ ^r[0-9]+$ ]]
                    export NEP_ACTUAL_A8B=1 NEP_ACTUAL_CELL=$cell
                    export NEP_CONTROLLED_ROUTING_SEED=101 NEP_ABLATION_FIXED_GRAD_ORDER=1
                    export NEP_CONTROLLED_REPLAY=0 NEP_CONTROLLED_ROUTE_UNIT=512 NEP_WORK_ALLOCATION=1
                    export ABLATION_MBS=2 ABLATION_GA=8 ABLATION_REDUCED_GA=8
                    export ABLATION_SEQLEN=8192
                    case "$cell" in
                        H32|H64|N32|N64)
                            export NVTE_BWD_LAYERNORM_SM_MARGIN=${cell:1}
                            export NEP_ABLATION_PARAM_ORDER=ascending
                            export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=5
                            [[ "$cell" != H* ]] || export GRAPH_CASES=healthy
                            ;;
                        B3|B5|B6)
                            export NEP_ABLATION_PARAM_ORDER=ascending
                            export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=${cell#B}
                            ;;
                        H0) export GRAPH_CASES=healthy ABLATION_SEQLEN=4096 NEP_ABLATION_PREFETCH_DEPTH=0 ;;
                        H1P) export GRAPH_CASES=healthy ABLATION_SEQLEN=4096 ;;
                        Q00) export ABLATION_SEQLEN=4096 NEP_ABLATION_PREFETCH_DEPTH=0 NEP_ABLATION_PARAM_RELEASE=consumer ;;
                        Q01) export ABLATION_SEQLEN=4096 NEP_ABLATION_PREFETCH_DEPTH=0 ;;
                        Q10) export ABLATION_SEQLEN=4096 NEP_ABLATION_PARAM_RELEASE=consumer ;;
                        Q11) export ABLATION_SEQLEN=4096 ;;
                        H4K) export GRAPH_CASES=healthy ABLATION_SEQLEN=4096 ;;
                        H2K) export GRAPH_CASES=healthy ABLATION_SEQLEN=2048 ;;
                        H8K|H2) export GRAPH_CASES=healthy ;;
                        N4K) export ABLATION_SEQLEN=4096 ;;
                        N2K) export ABLATION_SEQLEN=2048 ;;
                        N8K|M2) ;;
                        D4K) export ABLATION_SEQLEN=4096 NEP_ABLATION_PARAM_RELEASE=consumer ;;
                        D8K) export NEP_ABLATION_PARAM_RELEASE=consumer ;;
                        P4K) export ABLATION_SEQLEN=4096 NEP_ABLATION_PREFETCH_DEPTH=2 ;;
                        P8K) export NEP_ABLATION_PREFETCH_DEPTH=2 ;;
                        M1) export ABLATION_MBS=1 ABLATION_GA=16 ABLATION_REDUCED_GA=16 ;;
                        H4) export GRAPH_CASES=healthy ABLATION_MBS=4 ABLATION_GA=4 ABLATION_REDUCED_GA=4 ;;
                        M4) export ABLATION_MBS=4 ABLATION_GA=4 ABLATION_REDUCED_GA=4 ;;
                        H) export GRAPH_CASES=healthy ;;
                        H1) export GRAPH_CASES=healthy ABLATION_MBS=1 ABLATION_GA=16 ABLATION_REDUCED_GA=16 ;;
                        A) export NEP_ABLATION_PARAM_ORDER=ascending ;;
                        B) ;;
                        C) export NEP_ABLATION_PARAM_ORDER=ascending NEP_ABLATION_PARAM_LANES=2 ;;
                        D) export NEP_ABLATION_PARAM_LANES=2 ;;
                        E) export NEP_ABLATION_PLACEMENT=spread NEP_ABLATION_PARAM_ORDER=ascending ;;
                        F) export NEP_ABLATION_PLACEMENT=spread NEP_ABLATION_PARAM_ORDER=ascending NEP_ABLATION_PARAM_LANES=2 ;;
                        R) export ABLATION_REDUCED_GA=6 ;;
                        O) export ABLATION_REDUCED_GA=10 ;;
                        S) export ABLATION_MBS=1 ABLATION_GA=16 ABLATION_REDUCED_GA=16 ;;
                        *) exit 2 ;;
                    esac
                    second_ep=$GRAPH_FULL_EP
                    [[ "$GRAPH_CASES" != nep ]] || second_ep=$GRAPH_REDUCED_EP
                    export TRUE_GLOBAL_BATCH_SIZE=$((GRAPH_FULL_EP / 2 * ABLATION_MBS * ABLATION_GA + second_ep / 2 * ABLATION_MBS * ABLATION_REDUCED_GA))
                    ;;
                work_*)
                    IFS=_ read -r _ cell repeat <<< "$policy"
                    [[ "$repeat" =~ ^r[0-9]+$ ]]
                    export NEP_CONTROLLED_ROUTING_SEED=101 NEP_ABLATION_FIXED_GRAD_ORDER=1
                    export NEP_CONTROLLED_REPLAY=0 NEP_CONTROLLED_ROUTE_UNIT=512
                    export NEP_WORK_ALLOCATION=1 ABLATION_MBS=1 ABLATION_GA=4 ABLATION_REDUCED_GA=4
                    case "$cell" in
                        H1) export GRAPH_CASES=healthy ;;
                        H2) export GRAPH_CASES=healthy ABLATION_MBS=2 ABLATION_GA=2 ABLATION_REDUCED_GA=2 ;;
                        N1) ;;
                        N2) export ABLATION_MBS=2 ABLATION_GA=2 ABLATION_REDUCED_GA=2 ;;
                        R) export ABLATION_REDUCED_GA=3 ;;
                        O) export ABLATION_REDUCED_GA=5 ;;
                        *) exit 2 ;;
                    esac
                    second_ep=$GRAPH_FULL_EP
                    [[ "$GRAPH_CASES" != nep ]] || second_ep=$GRAPH_REDUCED_EP
                    export TRUE_GLOBAL_BATCH_SIZE=$((GRAPH_FULL_EP / 2 * ABLATION_MBS * ABLATION_GA + second_ep / 2 * ABLATION_MBS * ABLATION_REDUCED_GA))
                    ;;
                study_*)
                    IFS=_ read -r _ cell seed repeat mode <<< "$policy"
                    [[ "$cell" =~ ^[A-F]$ && "$seed" =~ ^s[0-9]+$ && "$repeat" =~ ^r[0-9]+$ ]]
                    [[ "$mode" == replay || "$mode" == training ]]
                    export NEP_CONTROLLED_ROUTING_SEED=${seed#s} NEP_ABLATION_FIXED_GRAD_ORDER=1
                    export NEP_CONTROLLED_REPLAY=0 NEP_ABLATION_PARAM_ORDER=ascending
                    [[ "$mode" != replay ]] || export NEP_CONTROLLED_REPLAY=1
                    case "$cell" in B|D) export NEP_ABLATION_PARAM_ORDER=waves ;; esac
                    case "$cell" in C|D|F) export NEP_ABLATION_PARAM_LANES=2 ;; esac
                    case "$cell" in E|F) export NEP_ABLATION_PLACEMENT=spread ;; esac
                    ;;
                reference|reference_repeat)
                    export CODE_REPO="$base/.worktrees/nep-param-destination-scatter-20260929" NEP_ABLATION_PARAM_LANES=0 ;;
                one|one_repeat) ;;
                two) export NEP_ABLATION_PARAM_LANES=2 ;;
                spread_two)
                    export NEP_ABLATION_PLACEMENT=spread NEP_ABLATION_PARAM_ORDER=ascending NEP_ABLATION_PARAM_LANES=2 NEP_ABLATION_FIXED_GRAD_ORDER=1 ;;
                random_compact|random_spread)
                    export NEP_ABLATION_PLACEMENT=${policy#random_} NEP_ABLATION_PARAM_ORDER=ascending NEP_ABLATION_FIXED_GRAD_ORDER=1
                    export NEP_CONTROLLED_ROUTING_SEED=${NEP_CONTROLLED_ROUTING_SEED:-101} ;;
                replay_A|replay_B|replay_C|replay_D|replay_E|replay_F)
                    export NEP_CONTROLLED_ROUTING_SEED=101 NEP_ABLATION_FIXED_GRAD_ORDER=1
                    export NEP_ABLATION_PARAM_ORDER=ascending
                    case "$policy" in
                        replay_B|replay_D) export NEP_ABLATION_PARAM_ORDER=waves ;;
                    esac
                    case "$policy" in
                        replay_C|replay_D|replay_F) export NEP_ABLATION_PARAM_LANES=2 ;;
                    esac
                    case "$policy" in
                        replay_E|replay_F) export NEP_ABLATION_PLACEMENT=spread ;;
                    esac
                    ;;
                *) exit 2 ;;
            esac
            ;;
        ablation_*)
            export CODE_REPO="${ABLATION_CODE_REPO:-$base/.worktrees/nep-placement-ablation-20260928}" NVTE_BWD_LAYERNORM_SM_MARGIN=32
            export NEP_ABLATION_PLACEMENT=current NEP_ABLATION_PARAM_ORDER=waves NEP_ABLATION_GRAD_ORDER=waves
            export NEP_ABLATION_PARAM_RELEASE=early NEP_ABLATION_GRAD_RELEASE=early
            export GRAPH_CASES=nep
            case "$implementation" in
                ablation_healthy*) export GRAPH_CASES=healthy ;;
                ablation_current_ascending) export NEP_ABLATION_PARAM_ORDER=ascending ;;
                ablation_spread_waves) export NEP_ABLATION_PLACEMENT=spread ;;
                ablation_spread_ascending) export NEP_ABLATION_PLACEMENT=spread NEP_ABLATION_PARAM_ORDER=ascending ;;
                ablation_grad_ascending) export NEP_ABLATION_GRAD_ORDER=ascending ;;
                ablation_spread_grad_ascending) export NEP_ABLATION_PLACEMENT=spread NEP_ABLATION_GRAD_ORDER=ascending ;;
                ablation_late_params) export NEP_ABLATION_PARAM_RELEASE=consumer ;;
                ablation_late_grads) export NEP_ABLATION_GRAD_RELEASE=finish ;;
                ablation_current_waves|ablation_current_repeat) ;;
                *) exit 2 ;;
            esac
            ;;
        reference) export CODE_REPO="$base/.worktrees/nep-graph-launch-order-20260921" ;;
        cleanup) export CODE_REPO="$base/.worktrees/nep-buffer-cleanup-20260921" ;;
        ag_overlap_margin) export CODE_REPO="$base/.worktrees/nep-ag-red-overlap-20260922" NVTE_BWD_LAYERNORM_SM_MARGIN=32 ;;
        param_waves_margin) export CODE_REPO="$base/.worktrees/nep-param-waves-20260922" NVTE_BWD_LAYERNORM_SM_MARGIN=32 ;;
        parallel) export CODE_REPO="$base/.worktrees/nep-param-owner-parallel-20260921" ;;
        parallel_margin) export CODE_REPO="$base/.worktrees/nep-param-owner-parallel-20260921" NVTE_BWD_LAYERNORM_SM_MARGIN=32 ;;
        *) exit 2 ;;
    esac
    export PYTHONPATH="$CODE_REPO:$base/scripts/nonuniform"
for GRAPH_CASE in ${GRAPH_CASES:-nep}; do
    export GRAPH_CASE
    for GRAPH_SCOPE in ${GRAPH_SCOPES:-moe}; do
        export GRAPH_SCOPE GRAPH_OUTPUT="$root/${implementation}_${GRAPH_CASE}_${GRAPH_SCOPE}"
        export GRAPH_MASTER_PORT=$((32210 + index))
        index=$((index + 1))
        mkdir -p "$GRAPH_OUTPUT"/{transport,placement,cupti_profile}
        git -C "$CODE_REPO" diff HEAD > "$GRAPH_OUTPUT/implementation.patch"
        sha256sum "$CODE_REPO/megatron/core/distributed/nonuniform_ep.py" > "$GRAPH_OUTPUT/runtime.sha256"
        echo "NATIVE_GRAPH_BEGIN $GRAPH_CASE $GRAPH_SCOPE $(date --iso-8601=seconds)"
        timeout --signal=TERM --kill-after=30s "${CLEANUP_CASE_TIMEOUT:-8m}" srun --nodes="$SLURM_NNODES" --ntasks="$((2 * GRAPH_FULL_EP))" --ntasks-per-node=4 \
            --kill-on-bad-exit=1 --mpi=none "${container[@]}" \
            bash "$runner" \
            > "$GRAPH_OUTPUT/driver.log" 2>&1
        echo "NATIVE_GRAPH_DONE $GRAPH_CASE $GRAPH_SCOPE $(date --iso-8601=seconds)"
        if [[ "$implementation" == controlled_actual_* ]]; then
            python3 "$CODE_REPO/tests/unit_tests/distributed/nep_actual_ablation.py" validate "$GRAPH_OUTPUT"
        fi
        if [[ "$implementation" == controlled_work_* ]]; then
            python3 "$CODE_REPO/tests/unit_tests/distributed/validate_nep_work_allocation.py" "$GRAPH_OUTPUT"
        fi
        if [[ "$implementation" == controlled_study_* ]]; then
            marker="$root/study-reference-${seed}-${mode}-${NEP_CONTROLLED_PROFILE}.txt"
            if [[ ! -f "$marker" ]]; then printf '%s\n' "$GRAPH_OUTPUT" > "$marker"; fi
            read -r study_reference < "$marker"
            python3 "$CODE_REPO/tests/unit_tests/distributed/validate_nep_study.py" \
                "$study_reference" "$GRAPH_OUTPUT" "$GRAPH_FULL_EP" "$GRAPH_REDUCED_EP" \
                "$mode" "$NEP_CONTROLLED_PROFILE" > "$GRAPH_OUTPUT/study-validation.log"
        fi
        if [[ "$implementation" == controlled_random_spread ]]; then
            python3 "$CODE_REPO/tests/unit_tests/distributed/validate_nep_randomized_controls.py" \
                "$root/controlled_random_compact_${GRAPH_CASE}_${GRAPH_SCOPE}" "$GRAPH_OUTPUT" \
                "$GRAPH_FULL_EP" "$GRAPH_REDUCED_EP" > "$root/randomized-controls-validation.json"
        fi
    done
done
done
