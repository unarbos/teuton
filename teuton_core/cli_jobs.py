"""Write-side helpers for the ``teuton-v3 send-job`` / ``submit-manifest`` CLI.

Single-shot, single-miner job emission. Reuses ``StreamingRunManager.emit_job``
under the hood (via the ``force_worker`` hook) so the manifest signing,
assignment grant, and bucket index plumbing all match what the orchestrator
emits.

This module is intentionally write-side only. Read-only views live in
:mod:`teuton_core.cli_views`.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from teuton_core import paths
from teuton_core.protocol import (
    ArtifactRef,
    AssignmentGrantV3,
    EncryptedAssignmentGrantV3,
    GraphRef,
    JobManifestV3,
    JobReceiptV3,
    ResourceRequirements,
    VerificationPolicy,
    WorkerIdentity,
)
from teuton_core.signatures import Signer
from teuton_core.wallet_crypto import (
    AssignmentEncryptor,
    DevAssignmentCrypto,
    Ed25519SealedBoxAssignmentCrypto,
)
from teuton_core.metagraph import BtcliMetagraphHotkeyResolver, MetagraphHotkeyResolver
from teuton_orchestrator.streaming import StreamingRunConfig, StreamingRunManager
from teuton_runtime.discovery import scan_bucket_discovery_records
from teuton_runtime.grants import broker_for_mode
from teuton_runtime.queue import OrchestratorQueue, QueueEntry
from teuton_runtime.storage import ObjectStore


_DEFAULT_ONESHOT_EPOCH_BASE = 9_000_000
_DEFAULT_ONESHOT_EPOCH_SPAN = 1_000_000


def _pick_oneshot_epoch() -> int:
    """Pick a synthetic epoch outside any range we know is in use.

    The orchestrator's stress mode starts at ``1_000_000``. Historical training
    runs live in the 0..max_epochs range. ``9_000_000+`` keeps one-shot
    manifests collision-free.
    """
    return _DEFAULT_ONESHOT_EPOCH_BASE + random.randint(0, _DEFAULT_ONESHOT_EPOCH_SPAN - 1)


def resolve_target_worker(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    hotkey: str,
    worker_id: str | None = None,
    heartbeat_ttl_sec: float | None = 120.0,
) -> WorkerIdentity:
    """Find the most recent heartbeating worker for ``hotkey`` on ``run_id``.

    Raises :class:`LookupError` with a helpful message when the hotkey has no
    recent heartbeat (the operator should fix that before trying to pin a
    job). When ``worker_id`` is supplied, prefer the matching record.
    """
    records = [
        r
        for r in scan_bucket_discovery_records(
            bucket,
            netuid=netuid,
            run_id=run_id,
            heartbeat_ttl_sec=heartbeat_ttl_sec,
        )
        if r.worker.hotkey_ss58 == hotkey
    ]
    if not records:
        raise LookupError(
            f"no fresh heartbeat for hotkey {hotkey} on run {run_id} (ttl={heartbeat_ttl_sec}s); "
            "check `teuton-v3 ls miners --run-id {run_id}` to confirm the miner is alive"
            .format(run_id=run_id)
        )
    if worker_id:
        for record in records:
            if record.worker.worker_id == worker_id:
                return record.worker
        raise LookupError(
            f"hotkey {hotkey} is heartbeating but no worker_id {worker_id} found "
            f"(have: {[r.worker.worker_id for r in records]})"
        )
    # Pick the freshest heartbeat.
    records.sort(key=lambda r: r.last_seen_unix, reverse=True)
    return records[0].worker


@dataclass
class SendJobResult:
    job_id: str
    run_id: str
    assigned_hotkey: str
    assigned_worker: str
    manifest: dict[str, Any]
    receipt: dict[str, Any] | None = None
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "assigned_hotkey": self.assigned_hotkey,
            "assigned_worker": self.assigned_worker,
            "manifest": self.manifest,
            "receipt": self.receipt,
            "timed_out": self.timed_out,
        }


def send_pipe_forward_job(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    hotkey: str,
    worker_id: str | None = None,
    stage: int = 0,
    mb: int = 0,
    epoch: int | None = None,
    owner_secret: str = "owner-dev-secret",
    owner_signer: Signer | None = None,
    grant_mode: str = "presigned",
    grant_ttl_sec: int = 86_400,
    assignment_crypto: str = "ed25519",
    assignment_secret: str = "teuton-dev-assignment",
    network: str = "finney",
    heartbeat_ttl_sec: float | None = 120.0,
) -> JobManifestV3:
    """Emit one ``pipe_forward`` manifest pinned to ``(hotkey, worker_id)``.

    Reuses an existing bootstrapped run on the bucket. The manifest is signed
    with ``owner_signer`` (or ``owner_secret``) and writes the same bucket
    keys that :meth:`StreamingRunManager.emit_job` writes, so the chosen
    miner picks it up on the next ``tick()``.
    """
    target_worker = resolve_target_worker(
        bucket,
        netuid=netuid,
        run_id=run_id,
        hotkey=hotkey,
        worker_id=worker_id,
        heartbeat_ttl_sec=heartbeat_ttl_sec,
    )

    # gpt_pipe writes its manifest config under the legacy ``runs/{run_id}/``
    # prefix (see teuton_legacy_v2/paths.manifest_config_key). We check that
    # path here so the helpful error fires before we hand off to
    # StreamingRunManager.bootstrap, which would otherwise raise a confusing
    # KeyError inside build_streaming_inputs.
    legacy_manifest_config_uri = bucket.uri_for_key(f"runs/{run_id}/manifest/config.json")
    if not bucket.exists(legacy_manifest_config_uri):
        raise FileNotFoundError(
            f"run {run_id} has no runs/{run_id}/manifest/config.json on the bucket. "
            "send-job requires an already-bootstrapped run (the orchestrator writes this during startup)."
        )

    manager = StreamingRunManager(
        bucket=bucket,
        config=StreamingRunConfig(
            netuid=netuid,
            run_id=run_id,
            task="gpt_pipe",
            max_epochs=1,
            owner_secret=owner_secret,
            owner_signer=owner_signer,
            grant_mode=grant_mode,
            grant_ttl_sec=grant_ttl_sec,
            assignment_secret=assignment_secret,
            assignment_crypto=assignment_crypto,
            network=network,
            discovery_backend="bucket",
            discovery_heartbeat_ttl_sec=heartbeat_ttl_sec,
            stress_emit=True,  # skips re-bootstrapping and pins weights to bootstrap epoch
            stress_skip_bootstrap_if_present=True,
        ),
    )
    manager.bootstrap()

    if stage < 0 or stage >= len(manager.stages):
        raise ValueError(
            f"stage {stage} out of range; run has {len(manager.stages)} stages (0..{len(manager.stages)-1})"
        )
    if mb < 0:
        raise ValueError(f"mb {mb} must be >= 0")

    # ``StreamingRunManager.emit_forward`` only writes the manifest; the graph
    # itself is uploaded by ``emit_epoch`` (which we don't call here). Ensure
    # the forward graph for this stage exists on the bucket before we publish
    # a manifest that references it. Without this the miner fetches the
    # GraphRef URI, gets a 404, silently skips the job, and the receipt never
    # lands.
    target_stage = manager.stages[stage]
    forward_graph = target_stage.forward_graph
    if forward_graph is not None:
        sha = forward_graph.graph_id()
        graph_uri = bucket.uri_for_key(paths.graph_key(netuid, run_id, sha))
        if not bucket.exists(graph_uri):
            bucket.put(graph_uri, forward_graph.to_canonical_json())

    chosen_epoch = epoch if epoch is not None else _pick_oneshot_epoch()
    manifest = manager.emit_forward(
        chosen_epoch,
        mb,
        target_stage,
        force_worker=target_worker,
    )
    return manifest


def submit_manifest_file(
    bucket: ObjectStore,
    *,
    manifest_path: str,
    owner_secret: str | None = None,
    owner_signer: Signer | None = None,
    grant_mode: str = "presigned",
    grant_ttl_sec: int = 86_400,
    assignment_crypto: str = "ed25519",
    assignment_secret: str = "teuton-dev-assignment",
    netuid: int | None = None,
    network: str = "finney",
    resign: bool = False,
) -> JobManifestV3:
    """Load a manifest JSON file and write it to the bucket as a single job.

    The manifest must already point at existing inputs/outputs and a graph
    that the bucket recognises. ``netuid`` defaults to the value parsed out
    of the manifest's resource layout when possible, otherwise the caller
    must pass it explicitly so the assignment grant lands at the right path.
    """
    with open(os.path.expanduser(manifest_path), "rb") as fh:
        payload = json.loads(fh.read().decode("utf-8"))
    manifest = JobManifestV3.from_dict(payload)
    if netuid is None:
        # Try to infer from the manifest's graph_ref URI, e.g.
        # ``s3://bucket/v3/netuid=3/runs/.../graphs/...``
        graph_uri = manifest.graph_ref.uri
        for part in graph_uri.split("/"):
            if part.startswith("netuid="):
                netuid = int(part[len("netuid=") :])
                break
    if netuid is None:
        raise ValueError("could not infer --netuid from manifest; pass --netuid explicitly")

    if resign or not manifest.owner_signature:
        secret_or_signer = owner_signer or owner_secret
        if secret_or_signer is None:
            raise ValueError("manifest is unsigned and no --owner-secret / wallet signer supplied")
        manifest = manifest.sign(secret_or_signer)

    # Manifest + assignment grant + queue entry. ``submit-manifest`` is a
    # one-off path so we open the queue, reconcile, add, flush, and stop --
    # no background thread.
    manifest_uri = bucket.uri_for_key(paths.job_manifest_key(netuid, manifest.run_id, manifest.job_id))
    bucket.put_json(manifest_uri, manifest.to_dict())

    grant_uri: str | None = None
    if grant_mode != "direct":
        grant_uri = _emit_assignment_grant_oneoff(
            bucket,
            manifest=manifest,
            netuid=netuid,
            grant_mode=grant_mode,
            grant_ttl_sec=grant_ttl_sec,
            assignment_crypto=assignment_crypto,
            assignment_secret=assignment_secret,
            network=network,
        )

    queue = OrchestratorQueue(
        bucket=bucket,
        netuid=netuid,
        run_id=manifest.run_id,
        role="train",
    )
    queue.reconcile_from_bucket()
    queue.add(QueueEntry.from_manifest(manifest, manifest_uri=manifest_uri, grant_uri=grant_uri))
    queue.flush(force=True)
    return manifest


def _emit_assignment_grant_oneoff(
    bucket: ObjectStore,
    *,
    manifest: JobManifestV3,
    netuid: int,
    grant_mode: str,
    grant_ttl_sec: int,
    assignment_crypto: str,
    assignment_secret: str,
    network: str,
) -> str | None:
    """Write the encrypted assignment grant for a one-off manifest.

    Mirrors :meth:`StreamingRunManager.emit_assignment_grant` but standalone
    so ``submit-manifest`` can run without instantiating the full manager.
    Returns the bucket URI of the written grant (``None`` when ``grant_mode``
    has no broker).
    """
    grant_broker = broker_for_mode(grant_mode, bucket)
    if grant_broker is None:
        return None
    crypto: AssignmentEncryptor = (
        Ed25519SealedBoxAssignmentCrypto()
        if assignment_crypto == "ed25519"
        else DevAssignmentCrypto(assignment_secret)
    )
    resolver: MetagraphHotkeyResolver | None = (
        BtcliMetagraphHotkeyResolver(netuid=netuid, network=network)
        if assignment_crypto == "ed25519"
        else None
    )

    now = int(time.time())
    receipt_uri = bucket.uri_for_key(
        paths.receipt_key(
            netuid,
            manifest.run_id,
            manifest.assigned_hotkey,
            manifest.job_id,
            manifest.attempt,
        )
    )
    grant = AssignmentGrantV3(
        job_id=manifest.job_id,
        run_id=manifest.run_id,
        assigned_hotkey=manifest.assigned_hotkey,
        input_gets=[
            grant_broker.get_grant(ref.uri, expires_in=grant_ttl_sec) for ref in manifest.inputs
        ],
        output_puts=[
            grant_broker.put_grant(ref.uri, expires_in=grant_ttl_sec) for ref in manifest.outputs
        ],
        receipt_put=grant_broker.put_grant(receipt_uri, expires_in=grant_ttl_sec),
        created_unix=now,
        expires_unix=now + int(grant_ttl_sec),
    )
    if resolver is not None:
        info = resolver.resolve(manifest.assigned_hotkey)
        encrypted = crypto.encrypt_for_hotkey(
            grant,
            recipient_hotkey=manifest.assigned_hotkey,
            recipient_uid=info.uid,
            metagraph_block=info.block,
            metagraph_hash=info.metagraph_hash,
            recipient_public_key=info.public_key,
        )
    else:
        encrypted = crypto.encrypt_for_hotkey(grant, recipient_hotkey=manifest.assigned_hotkey)
    grant_uri = bucket.uri_for_key(
        paths.assignment_key(netuid, manifest.run_id, manifest.job_id, manifest.assigned_hotkey)
    )
    bucket.put_json(grant_uri, encrypted.to_dict())
    return grant_uri


def wait_for_receipt(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    hotkey: str,
    job_id: str,
    attempt: int = 0,
    timeout_sec: float = 60.0,
    poll_interval: float = 1.0,
) -> JobReceiptV3 | None:
    """Block until ``receipt_key`` exists, or timeout. ``None`` on timeout."""
    receipt_uri = bucket.uri_for_key(
        paths.receipt_key(netuid, run_id, hotkey, job_id, attempt)
    )
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        try:
            if bucket.exists(receipt_uri):
                payload = bucket.get_json(receipt_uri)
                if isinstance(payload, dict):
                    return JobReceiptV3.from_dict(payload)
        except Exception:
            pass
        time.sleep(max(0.1, float(poll_interval)))
    return None


def render_send_job_summary(
    *,
    job_id: str,
    assigned_hotkey: str,
    assigned_worker: str,
    receipt: JobReceiptV3 | None,
    timed_out: bool,
    timeout_sec: float | None,
) -> str:
    head = f"emitted job_id={job_id} assigned={_short_ss58(assigned_hotkey)} worker={assigned_worker}"
    if receipt is None and timed_out:
        return f"{head}\nno receipt within {timeout_sec or 0:.0f}s"
    if receipt is None:
        return head
    age = max(0.0, time.time() - float(receipt.finished_unix or 0))
    bytes_read = int(receipt.claimed_bytes_read or 0)
    bytes_written = int(receipt.claimed_bytes_written or 0)
    return (
        f"{head}\n"
        f"receipt: compute={receipt.compute_sec:.3f}s read={_humanize_bytes(bytes_read)} "
        f"written={_humanize_bytes(bytes_written)} finished {age:.0f}s ago"
    )


def _short_ss58(value: str | None) -> str:
    if not value:
        return "-"
    if len(value) <= 12:
        return value
    return f"{value[:5]}...{value[-4:]}"


def _humanize_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024**2:
        return f"{n / 1024:.1f}KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f}MB"
    return f"{n / 1024**3:.2f}GB"


# ---------------------------------------------------------------------------
# Fleet health check: send one job to every live miner, in parallel.
# ---------------------------------------------------------------------------


@dataclass
class HealthCheckRow:
    hotkey_ss58: str
    worker_id: str
    gpu_class: str
    status: str  # "ok", "timeout", "skipped", "error"
    time_to_receipt_sec: float | None = None
    compute_sec: float | None = None
    bytes_read: int | None = None
    bytes_written: int | None = None
    job_id: str = ""
    error: str = ""
    last_seen_age_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotkey_ss58": self.hotkey_ss58,
            "worker_id": self.worker_id,
            "gpu_class": self.gpu_class,
            "status": self.status,
            "time_to_receipt_sec": self.time_to_receipt_sec,
            "compute_sec": self.compute_sec,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "job_id": self.job_id,
            "error": self.error,
            "last_seen_age_sec": self.last_seen_age_sec,
        }


def _send_and_wait_for_one(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    hotkey: str,
    owner_secret: str,
    owner_signer: Signer | None,
    grant_mode: str,
    grant_ttl_sec: int,
    assignment_crypto: str,
    assignment_secret: str,
    network: str,
    heartbeat_ttl_sec: float,
    per_miner_timeout_sec: float,
    receipt_poll_interval: float,
) -> HealthCheckRow:
    try:
        manifest = send_pipe_forward_job(
            bucket=bucket,
            netuid=netuid,
            run_id=run_id,
            hotkey=hotkey,
            owner_secret=owner_secret,
            owner_signer=owner_signer,
            grant_mode=grant_mode,
            grant_ttl_sec=grant_ttl_sec,
            assignment_crypto=assignment_crypto,
            assignment_secret=assignment_secret,
            network=network,
            heartbeat_ttl_sec=heartbeat_ttl_sec,
        )
    except LookupError as e:
        return HealthCheckRow(
            hotkey_ss58=hotkey,
            worker_id="",
            gpu_class="-",
            status="skipped",
            error=f"no fresh heartbeat: {e}",
        )
    except Exception as e:  # noqa: BLE001 - bucket / signing failures get surfaced
        return HealthCheckRow(
            hotkey_ss58=hotkey,
            worker_id="",
            gpu_class="-",
            status="error",
            error=f"{type(e).__name__}: {e}",
        )

    emit_t = time.time()
    receipt = wait_for_receipt(
        bucket=bucket,
        netuid=netuid,
        run_id=run_id,
        hotkey=manifest.assigned_hotkey,
        job_id=manifest.job_id,
        timeout_sec=per_miner_timeout_sec,
        poll_interval=receipt_poll_interval,
    )
    if receipt is None:
        return HealthCheckRow(
            hotkey_ss58=manifest.assigned_hotkey,
            worker_id=manifest.assigned_worker or "",
            gpu_class="-",
            status="timeout",
            job_id=manifest.job_id,
            error=f"no receipt within {per_miner_timeout_sec:.0f}s",
        )

    return HealthCheckRow(
        hotkey_ss58=manifest.assigned_hotkey,
        worker_id=manifest.assigned_worker or "",
        gpu_class="-",
        status="ok",
        time_to_receipt_sec=time.time() - emit_t,
        compute_sec=float(receipt.compute_sec or 0.0),
        bytes_read=int(receipt.claimed_bytes_read or 0),
        bytes_written=int(receipt.claimed_bytes_written or 0),
        job_id=manifest.job_id,
    )


def health_check(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    hotkeys: list[str] | None = None,
    owner_secret: str = "owner-dev-secret",
    owner_signer: Signer | None = None,
    grant_mode: str = "presigned",
    grant_ttl_sec: int = 86_400,
    assignment_crypto: str = "ed25519",
    assignment_secret: str = "teuton-dev-assignment",
    network: str = "finney",
    heartbeat_ttl_sec: float = 60.0,
    per_miner_timeout_sec: float = 300.0,
    receipt_poll_interval: float = 5.0,
    concurrency: int = 8,
    progress: Any = None,
) -> list[HealthCheckRow]:
    """Send one synthetic pipe_forward to each ``hotkey`` (or every live
    miner if ``hotkeys`` is ``None``) and return per-miner timing.

    Concurrent dispatch via :class:`concurrent.futures.ThreadPoolExecutor`
    keeps the sweep close to the slowest miner's latency rather than the
    sum across the fleet.
    """
    from teuton_runtime.discovery import scan_bucket_discovery_records

    if hotkeys is None:
        records = scan_bucket_discovery_records(
            bucket,
            netuid=netuid,
            run_id=run_id,
            heartbeat_ttl_sec=heartbeat_ttl_sec,
        )
        # GPU class + last-seen for the report; one entry per fresh hotkey.
        seen: dict[str, tuple[str, float]] = {}
        now = time.time()
        for record in records:
            cap = record.worker.capabilities or {}
            gpu_class = str(
                cap.get("gpu_class")
                or cap.get("gpu_model")
                or cap.get("gpu_name")
                or "-"
            )
            age = now - float(record.last_seen_unix or 0)
            prev = seen.get(record.worker.hotkey_ss58)
            if prev is None or age < prev[1]:
                seen[record.worker.hotkey_ss58] = (gpu_class, age)
        hotkeys = list(seen.keys())
    else:
        seen = {}

    rows: list[HealthCheckRow] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                _send_and_wait_for_one,
                bucket,
                netuid=netuid,
                run_id=run_id,
                hotkey=hk,
                owner_secret=owner_secret,
                owner_signer=owner_signer,
                grant_mode=grant_mode,
                grant_ttl_sec=grant_ttl_sec,
                assignment_crypto=assignment_crypto,
                assignment_secret=assignment_secret,
                network=network,
                heartbeat_ttl_sec=heartbeat_ttl_sec,
                per_miner_timeout_sec=per_miner_timeout_sec,
                receipt_poll_interval=receipt_poll_interval,
            ): hk
            for hk in hotkeys
        }
        done = 0
        total = len(futures)
        for future in concurrent.futures.as_completed(futures):
            hk = futures[future]
            row = future.result()
            # Decorate with GPU + last-seen if we discovered them.
            extra = seen.get(hk)
            if extra is not None:
                if row.gpu_class in ("", "-"):
                    row.gpu_class = extra[0]
                row.last_seen_age_sec = extra[1]
            rows.append(row)
            done += 1
            if progress is not None:
                progress(done, total, row)

    # Slow ones first.
    rows.sort(
        key=lambda r: (
            r.status not in ("ok",),
            -(r.time_to_receipt_sec or 0.0) if r.status == "ok" else 0.0,
        )
    )
    return rows


def render_health_check_table(rows: list[HealthCheckRow]) -> str:
    headers = ["HOTKEY", "WORKER", "GPU", "STATUS", "T_RECEIPT", "COMPUTE", "READ", "WRITTEN", "NOTE"]
    out_rows = []
    for r in rows:
        out_rows.append(
            [
                _short_ss58(r.hotkey_ss58),
                _short_worker(r.worker_id),
                r.gpu_class,
                r.status,
                f"{r.time_to_receipt_sec:.1f}s" if r.time_to_receipt_sec is not None else "-",
                f"{r.compute_sec:.3f}s" if r.compute_sec is not None else "-",
                _humanize_bytes(r.bytes_read or 0) if r.bytes_read is not None else "-",
                _humanize_bytes(r.bytes_written or 0) if r.bytes_written is not None else "-",
                r.error[:60] if r.error else "",
            ]
        )
    widths = [len(h) for h in headers]
    for row in out_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    for row in out_rows:
        lines.append(sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    # Aggregate stats
    ok_rows = [r for r in rows if r.status == "ok"]
    if ok_rows:
        durations = sorted([r.time_to_receipt_sec for r in ok_rows if r.time_to_receipt_sec is not None])
        n = len(durations)
        p50 = durations[n // 2] if n else 0
        p95 = durations[min(n - 1, int(0.95 * n))] if n else 0
        ok_count = len(ok_rows)
        lines.append("")
        lines.append(
            f"summary: {ok_count}/{len(rows)} ok, P50={p50:.1f}s P95={p95:.1f}s, "
            f"timeouts={sum(1 for r in rows if r.status == 'timeout')}, "
            f"errors={sum(1 for r in rows if r.status == 'error')}, "
            f"skipped={sum(1 for r in rows if r.status == 'skipped')}"
        )
    else:
        lines.append("")
        lines.append(
            f"summary: 0/{len(rows)} ok, "
            f"timeouts={sum(1 for r in rows if r.status == 'timeout')}, "
            f"errors={sum(1 for r in rows if r.status == 'error')}, "
            f"skipped={sum(1 for r in rows if r.status == 'skipped')}"
        )
    return "\n".join(lines)


def _short_worker(value: str | None) -> str:
    if not value:
        return "-"
    if "-" in value:
        return value.rsplit("-", 1)[-1]
    return _short_ss58(value)


# ---------------------------------------------------------------------------
# Stress stream: controlled-rate emission with metrics
# ---------------------------------------------------------------------------


@dataclass
class StreamSample:
    job_id: str
    hotkey: str
    emit_unix: float
    receipt_unix: float | None = None
    compute_sec: float | None = None
    timed_out: bool = False
    error: str = ""

    @property
    def latency_sec(self) -> float | None:
        if self.receipt_unix is None:
            return None
        return self.receipt_unix - self.emit_unix

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "hotkey": self.hotkey,
            "emit_unix": self.emit_unix,
            "receipt_unix": self.receipt_unix,
            "compute_sec": self.compute_sec,
            "latency_sec": self.latency_sec,
            "timed_out": self.timed_out,
            "error": self.error,
        }


@dataclass
class StreamReport:
    rate_per_min: float
    duration_sec: float
    emitted: int
    landed: int
    timed_out: int
    errored: int
    p50_latency_sec: float | None
    p95_latency_sec: float | None
    max_latency_sec: float | None
    samples: list[StreamSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate_per_min": self.rate_per_min,
            "duration_sec": self.duration_sec,
            "emitted": self.emitted,
            "landed": self.landed,
            "timed_out": self.timed_out,
            "errored": self.errored,
            "p50_latency_sec": self.p50_latency_sec,
            "p95_latency_sec": self.p95_latency_sec,
            "max_latency_sec": self.max_latency_sec,
            "samples": [s.to_dict() for s in self.samples],
        }


def stress_stream(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    rate_per_min: float,
    duration_sec: float,
    hotkeys: list[str],
    owner_secret: str = "owner-dev-secret",
    owner_signer: Signer | None = None,
    grant_mode: str = "presigned",
    grant_ttl_sec: int = 86_400,
    assignment_crypto: str = "ed25519",
    assignment_secret: str = "teuton-dev-assignment",
    network: str = "finney",
    receipt_timeout_sec: float = 300.0,
    progress: Any = None,
) -> StreamReport:
    """Emit ``rate_per_min`` jobs/minute for ``duration_sec`` seconds,
    round-robined across ``hotkeys``. After emission, wait up to
    ``receipt_timeout_sec`` for each pending job before declaring it stale.
    """
    if not hotkeys:
        raise ValueError("stress_stream requires at least one hotkey")
    interval = 60.0 / max(0.01, rate_per_min)
    emit_deadline = time.time() + duration_sec
    samples: list[StreamSample] = []
    hk_idx = 0
    emitted = 0
    errored = 0
    while time.time() < emit_deadline:
        hk = hotkeys[hk_idx % len(hotkeys)]
        hk_idx += 1
        t0 = time.time()
        try:
            manifest = send_pipe_forward_job(
                bucket=bucket,
                netuid=netuid,
                run_id=run_id,
                hotkey=hk,
                owner_secret=owner_secret,
                owner_signer=owner_signer,
                grant_mode=grant_mode,
                grant_ttl_sec=grant_ttl_sec,
                assignment_crypto=assignment_crypto,
                assignment_secret=assignment_secret,
                network=network,
                heartbeat_ttl_sec=120.0,
            )
            sample = StreamSample(
                job_id=manifest.job_id,
                hotkey=manifest.assigned_hotkey,
                emit_unix=t0,
            )
        except Exception as exc:  # noqa: BLE001
            sample = StreamSample(
                job_id=f"<emit_error-{int(t0)}>",
                hotkey=hk,
                emit_unix=t0,
                error=f"{type(exc).__name__}: {exc}",
            )
            errored += 1
        samples.append(sample)
        emitted += 1
        if progress is not None:
            progress("emit", sample, emitted, errored)
        # rate-pace; if we've already overshot, don't sleep
        now = time.time()
        sleep_for = (t0 + interval) - now
        if sleep_for > 0:
            time.sleep(sleep_for)

    # Wait phase: for each sample without a receipt, poll until receipt_timeout_sec.
    wait_deadline = time.time() + receipt_timeout_sec
    pending = [s for s in samples if not s.error and s.receipt_unix is None]
    while pending and time.time() < wait_deadline:
        for sample in list(pending):
            try:
                receipt = wait_for_receipt(
                    bucket=bucket,
                    netuid=netuid,
                    run_id=run_id,
                    hotkey=sample.hotkey,
                    job_id=sample.job_id,
                    timeout_sec=5.0,
                    poll_interval=1.0,
                )
            except Exception:
                receipt = None
            if receipt is not None:
                sample.receipt_unix = time.time()
                sample.compute_sec = float(receipt.compute_sec or 0.0)
                pending.remove(sample)
                if progress is not None:
                    progress("recv", sample, emitted, errored)
        if pending:
            time.sleep(2.0)

    for sample in pending:
        sample.timed_out = True

    landed_samples = [s for s in samples if s.receipt_unix is not None]
    landed = len(landed_samples)
    timed_out = sum(1 for s in samples if s.timed_out)
    latencies = sorted([s.latency_sec for s in landed_samples if s.latency_sec is not None])
    n = len(latencies)
    p50 = latencies[n // 2] if n else None
    p95 = latencies[min(n - 1, int(0.95 * n))] if n else None
    return StreamReport(
        rate_per_min=rate_per_min,
        duration_sec=duration_sec,
        emitted=emitted,
        landed=landed,
        timed_out=timed_out,
        errored=errored,
        p50_latency_sec=p50,
        p95_latency_sec=p95,
        max_latency_sec=latencies[-1] if latencies else None,
        samples=samples,
    )


def render_stream_report(report: StreamReport) -> str:
    lines = []
    lines.append(
        f"rate={report.rate_per_min:.1f}/min duration={report.duration_sec:.0f}s "
        f"emitted={report.emitted} landed={report.landed} "
        f"timed_out={report.timed_out} errored={report.errored}"
    )
    if report.landed:
        lines.append(
            f"latency: P50={report.p50_latency_sec:.1f}s "
            f"P95={report.p95_latency_sec:.1f}s "
            f"max={report.max_latency_sec:.1f}s"
        )
    landing_rate = report.landed / max(1.0, report.duration_sec) * 60.0
    emit_rate = report.emitted / max(1.0, report.duration_sec) * 60.0
    lines.append(f"effective rate: emit={emit_rate:.1f}/min land={landing_rate:.1f}/min")
    return "\n".join(lines)


# Re-export ``EncryptedAssignmentGrantV3`` for tests that want to round-trip
# the grant we just wrote without re-importing from teuton_core.protocol.
__all__ = [
    "SendJobResult",
    "HealthCheckRow",
    "StreamSample",
    "StreamReport",
    "send_pipe_forward_job",
    "submit_manifest_file",
    "wait_for_receipt",
    "health_check",
    "stress_stream",
    "render_send_job_summary",
    "render_health_check_table",
    "render_stream_report",
    "resolve_target_worker",
    "EncryptedAssignmentGrantV3",
]
