"""Optional early Proton bootstrap for ROCm profiling processes.

Python imports ``sitecustomize`` before the application module.  rocprofiler-sdk
must be configured before HIP/HSA registers its API tables, so the normal
``tokenspeed_kernel`` import can be too late in serving processes.
"""

import os


if os.environ.get("TOKENSPEED_KERNEL_PROFILE_EARLY", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    import tokenspeed_kernel  # noqa: F401
