"""Sequence-only extension of the frozen, validated single-knob launcher."""

import run_nep_single_knob_rank as launcher

launcher.PERTURBATIONS.update(
    seq2048=("--seq-length", "8192", "2048"),
    seq1024=("--seq-length", "8192", "1024"),
)
launcher.main()
