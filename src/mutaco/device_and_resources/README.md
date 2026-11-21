# Device and resources

[Device and precision resolution](device_and_precision_resolution.py) translates a
runtime request into the Torch device and dtype shared by MUTACO and its simulator.
An explicit CUDA request fails if the device is unavailable. Automatic selection
records why it fell back to the CPU.

[CPU parallelism provenance](cpu_parallelism_provenance.py) records processor
counts, Torch thread settings and thread-related environment overrides.
[Resource monitoring](resource_monitoring.py) reads allocator statistics on CUDA
and reports unavailable measurements explicitly on CPU. These observations leave
the scientific random streams and thread settings unchanged.

The original `mutaco.runtime` and `mutaco.observability` imports still reach
these same implementations. Python retains the historical module names for
serialized runtime requests, exceptions and monitoring records.
