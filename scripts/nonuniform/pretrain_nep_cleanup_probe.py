"""Run the unchanged profiling adapters with a caller-specified toy workload."""


def main():
    import pretrain_a8b_native_moe_graph as workload

    def train():
        from cuda_event_capture_adapter import CudaEventCapture
        from megatron.core.pipeline_parallel import schedules
        from pretrain_a8b_remaining_overheads import install_scopes
        from pretrain_profile_activity_probe import main as run

        install_scopes()
        CudaEventCapture.wrap(schedules, 'forward_step', 'microbatch.forward')
        CudaEventCapture.wrap(schedules, 'backward_step', 'microbatch.backward')
        run()

    workload.main = train
    from pretrain_a8b_graph_attribution import main as profile

    profile()


if __name__ == '__main__':
    main()
