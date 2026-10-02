"""Version-pinned sampling over several independent vLLM server replicas."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from moongazer_rl.async_rl.types import Lease, PublishRequest, PublishResult, adapter_name
from moongazer_rl.policy.base import GenerateRequest, GenerateResult
from moongazer_rl.policy.vllm_server_async import AsyncVllmServerPolicy

if TYPE_CHECKING:
    from moongazer_rl.trainer.unsloth_native.adapter_export import AdapterSnapshot


class LeaseUnavailable(ValueError):
    """No healthy replica can lease the pool's published version."""


class AsyncVllmPolicyPool:
    """Compose versioned vLLM policies without introducing a distributed runtime."""

    def __init__(
        self,
        base_urls: list[str],
        model: str,
        *,
        run_gen: str,
        export_adapter: Callable[[int], Path | str | AdapterSnapshot] | None = None,
        weight_version: int = 0,
        publish_sink: Callable[[PublishRequest], None] | None = None,
        unload_grace_s: float = 2.0,
        health_interval_s: float = 10.0,
        on_event: Callable[[str, dict], None] | None = None,
        adapter_export_root: str | Path | None = None,
        **policy_kwargs: Any,
    ) -> None:
        urls = [url.rstrip("/") for url in base_urls]
        if len(urls) < 2:
            raise ValueError("base_urls must contain at least 2 URLs")
        if len(set(urls)) != len(urls):
            raise ValueError("base_urls must not contain duplicates")
        if health_interval_s < 0:
            raise ValueError("health_interval_s must be >= 0")
        self.run_gen = run_gen
        self._prefix = str(policy_kwargs.get("lora_name_prefix", "policy"))
        self._export_adapter = export_adapter
        self._adapter_export_root = (
            Path(adapter_export_root) if adapter_export_root is not None else None
        )
        self._publish_sink = publish_sink
        self._published_version = int(weight_version)
        self._health_interval_s = float(health_interval_s)
        self._on_event = on_event or (lambda kind, data: None)
        self._lock = threading.RLock()
        self._maintenance_lock = threading.RLock()
        self._closed = False
        self._next_lease_id = 0
        self._leases: dict[int, int] = {}
        self._lease_counts: dict[int, int] = {}
        self._guards: dict[tuple[int, int], Lease] = {}
        # Publish owns these holds even when the last episode releases its lease.
        self._publish_guards: set[tuple[int, int]] = set()
        self._exports: dict[int, PublishRequest] = {}
        self._replicas: list[AsyncVllmServerPolicy] = []
        self._healthy = [True] * len(urls)
        self._last_health_check = [0.0] * len(urls)
        self._num_requests = [0] * len(urls)
        self._num_failures = [0] * len(urls)
        self._urls = urls
        for url in urls:
            self._replicas.append(
                AsyncVllmServerPolicy(
                    url,
                    model,
                    run_gen=run_gen,
                    export_adapter=None,
                    weight_version=weight_version,
                    publish_sink=None,
                    unload_grace_s=unload_grace_s,
                    owns_export_dir=False,
                    **policy_kwargs,
                )
            )

    @property
    def name(self) -> str:
        return "vllm-pool"

    @property
    def published_version(self) -> int:
        with self._lock:
            return self._published_version

    @property
    def weight_version(self) -> int:
        return self.published_version

    @property
    def served_model(self) -> str:
        return adapter_name(self.run_gen, self.published_version, self._prefix)

    def set_publish_sink(self, sink: Callable[[PublishRequest], None] | None) -> None:
        with self._lock:
            self._publish_sink = sink

    def _emit(self, kind: str, **data: Any) -> None:
        self._on_event(kind, {"mono": time.monotonic(), **data})

    def _loaded_versions(self, index: int) -> list[int]:
        return self._replicas[index].loaded_versions()

    def _mark_unhealthy(self, index: int, error: BaseException | str) -> None:
        with self._lock:
            if self._closed:
                # Shutdown cancels in-flight calls on every replica at once. Those failures
                # are the client's, not the server's.
                return
            changed = self._healthy[index]
            self._healthy[index] = False
            self._num_failures[index] += 1
        if changed:
            self._emit("replica_unhealthy", url=self._urls[index], error=str(error))

    def _check_health_due(self, *, force: bool = False) -> None:
        now = time.monotonic()
        due = []
        with self._lock:
            for index, checked in enumerate(self._last_health_check):
                if force or now - checked >= self._health_interval_s:
                    self._last_health_check[index] = now
                    due.append(index)
        if not due:
            return
        with ThreadPoolExecutor(max_workers=len(due)) as executor:
            futures = {executor.submit(self._replicas[i].check_server): i for i in due}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self._mark_unhealthy(index, exc)
                else:
                    with self._lock:
                        version = self._published_version
                        # A recovered replica must already be routable for live episodes.
                        # Otherwise publish will backfill/guard it before readmitting it.
                        recovered = (
                            not self._closed
                            and not self._healthy[index]
                            and version in self._loaded_versions(index)
                            and (
                                not self._lease_counts.get(version)
                                or (index, version) in self._guards
                            )
                        )
                        if recovered:
                            self._healthy[index] = True
                    if recovered:
                        self._emit("replica_recovered", url=self._urls[index])

    def check_server(self) -> dict[str, Any]:
        with self._maintenance_lock:
            self._check_health_due(force=True)
            with self._lock:
                if not any(self._healthy):
                    raise RuntimeError("no healthy vLLM replicas")
            return self.stats()

    def _export_version(self, version: int) -> PublishRequest:
        with self._lock:
            if self._closed:
                raise RuntimeError("policy is closed")
            if version < self._published_version:
                raise ValueError(f"cannot export retired policy version {version}")
            existing = self._exports.get(version)
        if existing is not None and (existing.path / "adapter_config.json").is_file():
            return existing
        if self._export_adapter is None:
            raise RuntimeError("async policy requires export_adapter")
        exported: Any = self._export_adapter(version)
        snapshot = None
        if isinstance(exported, str | Path):
            path = Path(exported).resolve()
            if not (path / "adapter_config.json").is_file():
                raise FileNotFoundError(f"no adapter_config.json under exported adapter {path}")
        else:
            # Snapshot mode (adapter_export.py): the learner thread only cloned the LoRA
            # tensors; publish() writes them once, before any replica loads the directory.
            from moongazer_rl.trainer.unsloth_native.adapter_export import AdapterSnapshot

            if not isinstance(exported, AdapterSnapshot):
                raise TypeError("export_adapter must return a path or AdapterSnapshot")
            if exported.version != version:
                raise ValueError(
                    f"adapter snapshot version {exported.version} does not match {version}"
                )
            if self._adapter_export_root is None:
                raise RuntimeError("snapshot export requires adapter_export_root")
            path = (self._adapter_export_root / f"v{version}").resolve()
            snapshot = exported
        request = PublishRequest(version, path, snapshot=snapshot)
        with self._lock:
            if any(v != version and req.path == path for v, req in self._exports.items()):
                raise ValueError(f"adapter export path is shared by multiple versions: {path}")
            self._exports[version] = request
        return request

    def set_weight_version(self, version: int) -> None:
        with self._lock:
            sink = self._publish_sink
        if sink is None:
            raise RuntimeError("publish sink is not set")
        sink(self._export_version(int(version)))

    async def sync_weights(self, version: int) -> None:
        return None

    def lease(self) -> Lease:
        """Acquire the published version using state operations only; never wait for publish."""
        acquired: dict[tuple[int, int], Lease] = {}
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("policy is closed")
                version = self._published_version
                candidates = [
                    i
                    for i, healthy in enumerate(self._healthy)
                    if healthy and version in self._loaded_versions(i)
                ]
                count = self._lease_counts.get(version, 0)
                if count == 0:
                    for index in candidates:
                        key = (index, version)
                        if key not in self._guards:
                            try:
                                acquired[key] = self._replicas[index].lease_version(version)
                            except ValueError:
                                # A retiring replica is not an available lease target.
                                continue
                if not any(
                    (i, version) in self._guards or (i, version) in acquired for i in candidates
                ):
                    raise LeaseUnavailable(
                        f"policy version {version} is not loaded on any healthy replica"
                    )
                lease = Lease(version, self._next_lease_id)
                self._next_lease_id += 1
                self._guards.update(acquired)
                self._leases[lease.lease_id] = version
                self._lease_counts[version] = count + 1
                return lease
        except BaseException:
            # release() may schedule cleanup; keep it outside the pool state lock.
            for (index, _), guard in acquired.items():
                self._replicas[index].release(guard)
            raise

    def release(self, lease: Lease) -> None:
        guards: list[tuple[AsyncVllmServerPolicy, Lease]] = []
        with self._lock:
            version = self._leases.get(lease.lease_id)
            if version is None or version != lease.version:
                return
            del self._leases[lease.lease_id]
            count = self._lease_counts[version] - 1
            if count:
                self._lease_counts[version] = count
            else:
                del self._lease_counts[version]
                for key in [
                    key
                    for key in self._guards
                    if key[1] == version and key not in self._publish_guards
                ]:
                    guards.append((self._replicas[key[0]], self._guards.pop(key)))
        for replica, guard in guards:
            replica.release(guard)

    def _route_index(self, request: GenerateRequest, version: int) -> int:
        with self._lock:
            leased = self._lease_counts.get(version, 0) > 0
            candidates = [
                i
                for i, healthy in enumerate(self._healthy)
                if healthy
                and version in self._loaded_versions(i)
                and (not leased or (i, version) in self._guards)
            ]
        if not candidates:
            raise ValueError(f"policy version {version} is not loaded on any replica")
        affinity = request.session_id or request.request_id or ""
        digest = hashlib.sha256(affinity.encode("utf-8")).digest()
        return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        version = (
            self.published_version if request.policy_version is None else request.policy_version
        )
        index = self._route_index(request, version)
        with self._lock:
            self._num_requests[index] += 1
        try:
            return await self._replicas[index].generate(request)
        except httpx.HTTPError as exc:
            # A cancelled rollout (shutdown, evaluator stop) surfaces as ReadError/CloseError
            # with an empty message; that is the client's doing, not the replica's. Connection
            # failures and protocol errors still take the replica out of rotation until the
            # next health check.
            if not self._closed and not isinstance(
                exc, httpx.ReadError | httpx.WriteError | httpx.CloseError
            ):
                self._mark_unhealthy(index, exc)
            raise
        except RuntimeError as exc:
            # AsyncVllmServerPolicy.generate raises "vLLM /v1/completions returned <status>"
            # for any non-200 answer. A 4xx is this request's problem (e.g. a prompt over
            # max_model_len), not the replica's; only a server-side failure drops the replica.
            if "returned 5" in str(exc):
                self._mark_unhealthy(index, exc)
            raise

    def _ensure_guard(self, index: int, version: int, *, force: bool = False) -> bool:
        with self._lock:
            needed = force or self._lease_counts.get(version, 0) > 0
            present = (index, version) in self._guards
            if needed and not present:
                self._guards[index, version] = self._replicas[index].lease_version(version)
            if force:
                self._publish_guards.add((index, version))
            return force

    def publish(self, request: PublishRequest) -> PublishResult:
        with self._maintenance_lock:
            snapshot = request.snapshot
            request = PublishRequest(
                request.version,
                request.path.resolve(),
                request.requested_at,
                request.requested_mono,
            )
            export_s = 0.0
            if snapshot is not None and not (request.path / "adapter_config.json").is_file():
                from moongazer_rl.trainer.unsloth_native.adapter_export import write_adapter

                started = time.monotonic()
                write_adapter(snapshot, request.path)
                export_s = time.monotonic() - started
            if not (request.path / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"no adapter_config.json under exported adapter {request.path}"
                )
            with self._lock:
                if self._closed:
                    raise RuntimeError("policy is closed")
                if request.version < self._published_version:
                    raise ValueError(f"cannot publish older policy version {request.version}")
                self._exports[request.version] = request
            self._check_health_due()
            failed: set[int] = set()
            target_ready: set[int] = set()
            unloaded: set[int] = set()
            cleanup_errors: list[str] = []
            load_times: list[float] = []
            temporary_guards: set[tuple[int, int]] = set()
            try:
                with self._lock:
                    old_version = self._published_version
                    for index, healthy in enumerate(self._healthy):
                        if healthy and old_version in self._loaded_versions(index):
                            try:
                                self._ensure_guard(index, old_version, force=True)
                            except ValueError:
                                # The replica is already retiring P_old (it committed a
                                # publish the pool never flipped to); it cannot serve P_old
                                # leases, so there is nothing to protect there.
                                continue
                            temporary_guards.add((index, old_version))
                    leased_versions = {
                        version for version, count in self._lease_counts.items() if count > 0
                    }
                    export_requests = {
                        version: exported
                        for version, exported in self._exports.items()
                        if version == request.version or version in leased_versions
                    }
                for version, version_request in sorted(export_requests.items()):
                    if not (version_request.path / "adapter_config.json").is_file():
                        continue
                    participants = []
                    for index in range(len(self._replicas)):
                        if index in failed:
                            continue
                        loaded = self._loaded_versions(index)
                        if version == request.version or version not in loaded:
                            participants.append(index)
                    if not participants:
                        continue
                    with ThreadPoolExecutor(max_workers=len(participants)) as executor:
                        futures = {
                            executor.submit(self._replicas[i].publish, version_request): i
                            for i in participants
                        }
                        for future in as_completed(futures):
                            index = futures[future]
                            try:
                                result = future.result()
                            except Exception as exc:
                                failed.add(index)
                                self._mark_unhealthy(index, exc)
                            else:
                                load_s = result.load_s
                                load_times.append(load_s)
                                cleanup_errors.extend(
                                    f"{self._urls[index]}: {error}"
                                    for error in result.cleanup_errors
                                )
                                self._emit(
                                    "publish_replica_finished",
                                    url=self._urls[index],
                                    version=version,
                                    load_s=load_s,
                                )
                                with self._lock:
                                    recovering = not self._healthy[index]
                                try:
                                    if self._ensure_guard(index, version, force=recovering):
                                        temporary_guards.add((index, version))
                                except ValueError:
                                    # The replica committed this version before its recovery
                                    # guard could be established. Skip its guard without
                                    # aborting publication on the other replicas.
                                    continue
                                if version == request.version:
                                    target_ready.add(index)
                for index in range(len(self._replicas)):
                    if index not in failed and request.version in self._loaded_versions(index):
                        target_ready.add(index)
                with self._lock:
                    if self._closed:
                        raise RuntimeError("policy is closed")
                if not target_ready:
                    raise RuntimeError(f"publish version {request.version} failed on all replicas")
                recovered: list[int] = []
                with self._lock:
                    if self._closed:
                        raise RuntimeError("policy is closed")
                    recovered = [index for index in target_ready if not self._healthy[index]]
                    for index in target_ready:
                        self._healthy[index] = True
                    self._published_version = request.version
                for index in recovered:
                    self._emit("replica_recovered", url=self._urls[index])
            finally:
                releases: list[tuple[AsyncVllmServerPolicy, Lease]] = []
                with self._lock:
                    for key in temporary_guards:
                        self._publish_guards.discard(key)
                        if not self._lease_counts.get(key[1]):
                            guard = self._guards.pop(key, None)
                            if guard is not None:
                                releases.append((self._replicas[key[0]], guard))
                for replica, guard in releases:
                    replica.release(guard)
            unloaded.update(self.prune())
            return PublishResult(
                request.version,
                adapter_name(self.run_gen, request.version, self._prefix),
                max(load_times, default=0.0),
                sorted(unloaded),
                cleanup_errors,
                export_s=export_s,
            )

    def _cleanup_previous(self, index: int) -> None:
        self._replicas[index].cleanup_stale_adapters()

    def initial_publish(self, version: int) -> PublishResult:
        with ThreadPoolExecutor(max_workers=len(self._replicas)) as executor:
            futures = {
                executor.submit(self._cleanup_previous, i): i for i in range(len(self._replicas))
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self._mark_unhealthy(index, exc)
        return self.publish(self._export_version(int(version)))

    def _drop_missing_exports(self) -> None:
        with self._lock:
            exports = dict(self._exports)
        missing = {version for version, request in exports.items() if not request.path.exists()}
        with self._lock:
            for version in missing:
                if self._exports.get(version) == exports[version] and not self._lease_counts.get(
                    version
                ):
                    del self._exports[version]

    def prune(self) -> list[int]:
        with self._maintenance_lock:
            self._check_health_due()
            for replica in self._replicas:
                replica.prune()
            with self._lock:
                candidates = {
                    version: exported
                    for version, exported in self._exports.items()
                    if version < self._published_version and not self._lease_counts.get(version)
                }
            removed: list[int] = []
            for version, exported in candidates.items():
                if any(replica.is_version_busy(version) for replica in self._replicas):
                    continue
                shutil.rmtree(exported.path, ignore_errors=True)
                if exported.path.exists():
                    continue
                with self._lock:
                    still_unused = not self._lease_counts.get(version)
                    if still_unused and self._exports.get(version) == exported:
                        del self._exports[version]
                        removed.append(version)
            self._drop_missing_exports()
            return sorted(removed)

    def registered_versions(self) -> dict[int, int]:
        totals: dict[int, int] = {}
        guard_counts: dict[int, int] = {}
        with self._lock:
            lease_counts = dict(self._lease_counts)
            export_versions = set(self._exports)
            for _, version in self._guards:
                guard_counts[version] = guard_counts.get(version, 0) + 1
        for replica in self._replicas:
            for version, count in replica.registered_versions().items():
                totals[version] = totals.get(version, 0) + count
        for version, count in guard_counts.items():
            totals[version] = totals.get(version, 0) - count
        for version, count in lease_counts.items():
            totals[version] = totals.get(version, 0) + count
        for version in export_versions:
            totals.setdefault(version, lease_counts.get(version, 0))
        return dict(sorted(totals.items()))

    def forget(self, version: int) -> bool:
        with self._lock:
            if version == self._published_version or self._lease_counts.get(version):
                return False
        present: list[AsyncVllmServerPolicy] = []
        for replica in self._replicas:
            if replica.is_version_busy(version):
                return False
            if version in replica.registered_versions():
                present.append(replica)
        for replica in present:
            if not replica.forget(version):
                return False
        with self._lock:
            return self._exports.pop(version, None) is not None

    def stats(self) -> dict[str, Any]:
        replicas = []
        with self._lock:
            healthy = list(self._healthy)
            requests = list(self._num_requests)
            failures = list(self._num_failures)
            guards_per_replica = [
                sum(1 for guarded_index, _ in self._guards if guarded_index == index)
                for index in range(len(self._replicas))
            ]
        for index, replica in enumerate(self._replicas):
            loaded = replica.loaded_versions()
            registered = replica.registered_versions()
            in_flight = sum(registered.values()) - guards_per_replica[index]
            replicas.append(
                {
                    "url": self._urls[index],
                    "healthy": healthy[index],
                    "loaded_versions": loaded,
                    "in_flight": in_flight,
                    "num_requests": requests[index],
                    "num_failures": failures[index],
                }
            )
        return {"replicas": replicas}

    async def close(self) -> None:
        with self._lock:
            self._closed = True

        def wait_for_publish() -> None:
            with self._maintenance_lock:
                pass

        async def finish() -> None:
            await asyncio.to_thread(wait_for_publish)
            await asyncio.gather(*(replica.close() for replica in self._replicas))

        worker = asyncio.create_task(finish())
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            await worker
            raise


__all__ = ["AsyncVllmPolicyPool", "LeaseUnavailable"]
