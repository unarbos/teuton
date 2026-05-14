# Scaling Lessons From V2

The v2 experiments showed that the shared bucket protocol works across heterogeneous global fleets, but S3 round-trip latency dominates small jobs. Synchronous rounds hit a utilization ceiling because forward, inner, reduce, outer, and eval phases gate each other. Streaming improved worker participation and compute share, while still exposing bucket latency as the primary bottleneck.

Operational lessons carried into v3:

- Workers should publish capability metadata such as GPU class, device count, host, and bucket RTT.
- Orchestrators need startup grace, stage or UB coverage checks, and retry/eviction for stale workers.
- GPU workers should self-probe before heartbeating so broken devices do not enter the assignment pool.
- Telemetry should separate wallclock, busy time, compute time, IO time, bytes, active workers, and cost.
- For active inter-stage communication, S3 is acceptable for correctness and replayability but not the final low-latency substrate.
