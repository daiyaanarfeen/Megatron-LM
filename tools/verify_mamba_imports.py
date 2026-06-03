#!/usr/bin/env python3
"""Verify Mamba benchmark runtime imports."""

import causal_conv1d
import mamba_ssm
import megatron

print("mamba_ssm", getattr(mamba_ssm, "__version__", "unknown"))
print("causal_conv1d", getattr(causal_conv1d, "__version__", "unknown"))
print("megatron import ok", megatron.__name__)
