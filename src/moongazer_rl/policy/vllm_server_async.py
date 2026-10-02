"""Version-pinned sampling with exports independent of vLLM administration."""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from moongazer_rl.async_rl.types import Lease, PublishRequest, PublishResult, adapter_name
from moongazer_rl.policy.base import GenerateRequest, GenerateResult
from moongazer_rl.policy.vllm_server import VllmServerPolicy

if TYPE_CHECKING:
    from moongazer_rl.trainer.unsloth_native.adapter_export import AdapterSnapshot


@dataclass
class _VersionState:
    name: str
    path: Path
    leases: int = 0
    in_flight: int = 0
    zero_since_mono: float | None = None
    loaded: bool = False


class LeasedAdapterRegistry:
    """Share lifetime bookkeeping with the CPU double so race tests exercise the same rules.

    Lock order is maintenance -> registry. Registry sections only manipulate state; neither
    lock may acquire the learner model lock. The learner never takes the maintenance lock.
    """

    def _init_registry(
        self,
        *,
        run_gen: str,
        weight_version: int,
        prefix: str,
        export_adapter: Callable[[int], Path | str | AdapterSnapshot] | None,
        publish_sink: Callable[[PublishRequest], None] | None,
        unload_grace_s: float,
        adapter_export_root: Path | None = None,
        owns_export_dir: bool = True,
    ) -> None:
        if unload_grace_s < 0:
            raise ValueError("unload_grace_s must be >= 0")
        self.run_gen = run_gen
        self._prefix = prefix
        self._published_version = int(weight_version)
        self._export_adapter = export_adapter
        self._publish_sink = publish_sink
        self._adapter_export_root = adapter_export_root
        self.unload_grace_s = float(unload_grace_s)
        self._owns_export_dir = bool(owns_export_dir)
        self._registry: dict[int, _VersionState] = {}
        self._registry_lock = threading.Lock()
        # Only maintenance workers take this lock; the learner never waits for HTTP.
        self._maintenance_lock = threading.RLock()
        self._retiring: set[int] = set()
        self._exporting: set[int] = set()
        self._leases: dict[int, int] = {}
        self._next_lease_id = 0
        self._closed = False
        self._requests_drained = threading.Event()
        self._requests_drained.set()
        self.last_cleanup_errors: list[str] = []
        self._cleanup_tasks: set[asyncio.Task[None]] = set()

    @property
    def published_version(self) -> int:
        with self._registry_lock:
            return self._published_version

    @property
    def weight_version(self) -> int:
        return self.published_version

    @property
    def served_model(self) -> str:
        return adapter_name(self.run_gen, self.published_version, self._prefix)

    def set_publish_sink(self, sink: Callable[[PublishRequest], None] | None) -> None:
        with self._registry_lock:
            self._publish_sink = sink

    def lease(self) -> Lease:
        return self.lease_version(None)

    def lease_version(self, version: int | None) -> Lease:
        """Lease ``version`` (``None`` = the published version). A pool guards a version on
        every replica that holds it, so the guard must pin *that* version, not whatever the
        replica happens to publish at the moment (it may already be ahead of the pool)."""
        with self._registry_lock:
            if self._closed:
                raise RuntimeError("policy is closed")
            if version is None:
                version = self._published_version
            state = self._registry.get(version)
            if state is None or not state.loaded or version in self._retiring:
                raise ValueError(f"policy version {version} is not loaded")
            lease = Lease(version=version, lease_id=self._next_lease_id)
            self._next_lease_id += 1
            self._leases[lease.lease_id] = version
            state.leases += 1
            state.zero_since_mono = None
            return lease

    def release(self, lease: Lease) -> None:
        with self._registry_lock:
            version = self._leases.get(lease.lease_id)
            if version is None or version != lease.version:
                return
            del self._leases[lease.lease_id]
            state = self._registry[version]
            state.leases -= 1
            zero = state.leases + state.in_flight == 0
            if zero:
                state.zero_since_mono = time.monotonic()
            cleanup = zero and version < self._published_version
        if cleanup:
            self._schedule_prune()

    def _begin_generate(self, request: GenerateRequest) -> tuple[int, str]:
        with self._registry_lock:
            if self._closed:
                raise RuntimeError("policy is closed")
            version = (
                self._published_version
                if request.policy_version is None
                else request.policy_version
            )
            state = self._registry.get(version)
            if state is None or not state.loaded or version in self._retiring:
                raise ValueError(f"policy version {version} is not loaded")
            state.in_flight += 1
            self._requests_drained.clear()
            state.zero_since_mono = None
            return version, state.name

    def _end_generate(self, version: int) -> None:
        with self._registry_lock:
            state = self._registry[version]
            state.in_flight -= 1
            if not any(state.in_flight for state in self._registry.values()):
                self._requests_drained.set()
            zero = state.leases + state.in_flight == 0
            if zero:
                state.zero_since_mono = time.monotonic()
            cleanup = zero and version < self._published_version
        if cleanup:
            self._schedule_prune()

    def _schedule_prune(self) -> None:
        """Release runs on the loop; grace and blocking unload belong in a separate task."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Startup/tests may release synchronously; the next publish/prune retries cleanup.
            return
        if self._closed:
            return

        async def cleanup() -> None:
            if self.unload_grace_s:
                await asyncio.sleep(self.unload_grace_s)
            worker = asyncio.create_task(asyncio.to_thread(self.prune))
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Cancelling a to_thread future does not stop its HTTP operation.
                await worker
                raise

        task = loop.create_task(cleanup())
        self._cleanup_tasks.add(task)

        def cleanup_done(done: asyncio.Task[None]) -> None:
            self._cleanup_tasks.discard(done)
            if done.cancelled():
                return
            exc = done.exception()
            if exc is not None:
                self.last_cleanup_errors.append(f"cleanup task: {exc!r}")

        task.add_done_callback(cleanup_done)

    def registered_versions(self) -> dict[int, int]:
        with self._registry_lock:
            return {v: s.leases + s.in_flight for v, s in self._registry.items()}

    def loaded_versions(self) -> list[int]:
        """Return loaded adapter versions without exposing registry internals."""
        with self._registry_lock:
            return sorted(version for version, state in self._registry.items() if state.loaded)

    def is_version_busy(self, version: int) -> bool:
        """Whether a version is loaded, leased, generating, retiring, or published."""
        with self._registry_lock:
            state = self._registry.get(version)
            return bool(
                version == self._published_version
                or version in self._retiring
                or (
                    state is not None
                    and (state.loaded or state.leases != 0 or state.in_flight != 0)
                )
            )

    def forget(self, version: int) -> bool:
        """Drop only an export that cannot be serving or become the lease target."""
        with self._registry_lock:
            state = self._registry.get(version)
            if (
                state is None
                or state.leases != 0
                or state.in_flight != 0
                or state.loaded
                or version == self._published_version
            ):
                return False
            del self._registry[version]
            return True

    def _export_version(self, version: int) -> PublishRequest:
        with self._registry_lock:
            if self._closed:
                raise RuntimeError("policy is closed")
            if version < self._published_version or version in self._retiring:
                raise ValueError(f"cannot export retired policy version {version}")
            state = self._registry.get(version)
            existing_path = state.path if state is not None else None
            if version in self._exporting:
                raise RuntimeError(f"policy version {version} is already being exported")
        # This filesystem pre-check is lock-free and safe because only the learner thread calls
        # this method while holding the model lock. The next in-lock re-check is defensive.
        if existing_path is not None and (existing_path / "adapter_config.json").is_file():
            # Never overwrite files that vLLM may still be reading for a live adapter.
            return PublishRequest(version, existing_path)
        with self._registry_lock:
            if version in self._exporting:
                raise RuntimeError(f"policy version {version} is already being exported")
            self._exporting.add(version)
        try:
            if self._export_adapter is None:
                raise RuntimeError("async policy requires export_adapter")
            exported: Any = self._export_adapter(version)
            snapshot = None
            if isinstance(exported, str | Path):
                path = Path(exported).resolve()
                if not (path / "adapter_config.json").is_file():
                    raise FileNotFoundError(f"no adapter_config.json under exported adapter {path}")
            else:
                from moongazer_rl.trainer.unsloth_native.adapter_export import (
                    AdapterSnapshot,
                )

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
            with self._registry_lock:
                if any(v != version and s.path == path for v, s in self._registry.items()):
                    raise ValueError(f"adapter export path is shared by multiple versions: {path}")
                state = self._registry.get(version)
                if state is None:
                    self._registry[version] = _VersionState(
                        adapter_name(self.run_gen, version, self._prefix),
                        path,
                        zero_since_mono=time.monotonic(),
                    )
                else:
                    state.path = path
            return PublishRequest(version, path, snapshot=snapshot)
        finally:
            with self._registry_lock:
                self._exporting.discard(version)

    def set_weight_version(self, version: int) -> None:
        """Export under the learner's model lock; enqueue without waiting for server loading."""
        with self._registry_lock:
            sink = self._publish_sink
        if sink is None:
            raise RuntimeError("publish sink is not set")
        request = self._export_version(int(version))
        sink(request)

    async def sync_weights(self, version: int) -> None:
        """The publisher owns loading; a synchronous-path sync must not create a barrier."""

    def _load_adapter(self, state: _VersionState) -> None:
        raise NotImplementedError

    def _unload_adapter(self, state: _VersionState) -> None:
        raise NotImplementedError

    def publish(self, request: PublishRequest) -> PublishResult:
        """Serialize maintenance, commit after load, and keep cleanup failures post-commit."""
        with self._maintenance_lock:
            path = request.path.resolve()
            with self._registry_lock:
                version = request.version
                if self._closed:
                    raise RuntimeError("policy is closed")
                if version < self._published_version:
                    raise ValueError(f"cannot publish older policy version {version}")
                if version in self._exporting:
                    raise RuntimeError(f"policy version {version} is still being exported")
                state = self._registry.get(version)
                if state is not None and state.path != path:
                    raise ValueError(f"policy version {version} already has another export path")
                if state is None:
                    if any(s.path == path for s in self._registry.values()):
                        raise ValueError(f"adapter export path is shared: {path}")
                    state = _VersionState(
                        adapter_name(self.run_gen, version, self._prefix),
                        path,
                        zero_since_mono=time.monotonic(),
                    )
                    self._registry[version] = state
                loaded = state.loaded
            export_s = 0.0
            if request.snapshot is not None:
                from moongazer_rl.trainer.unsloth_native.adapter_export import write_adapter

                started = time.monotonic()
                write_adapter(request.snapshot, path)
                export_s = time.monotonic() - started
            load_s = 0.0
            if not loaded:
                try:
                    self._unload_adapter(state)
                    started = time.monotonic()
                    self._load_adapter(state)
                    load_s = time.monotonic() - started
                except Exception as exc:
                    raise RuntimeError(f"publish version {version} failed: {exc}") from exc
                with self._registry_lock:
                    state.loaded = True
                    if self._closed:
                        raise RuntimeError("policy is closed")
                    self._published_version = version
            # A repeated successful request must not unload an adapter with active readers.
            result = PublishResult(version, state.name, load_s, export_s=export_s)
            try:
                result.unloaded_versions = self.prune()
                result.cleanup_errors = list(self.last_cleanup_errors)
            except Exception as exc:
                # Preserve the commit even if an overridden cleanup implementation fails.
                result.cleanup_errors = [f"prune: {exc}"]
                self.last_cleanup_errors = list(result.cleanup_errors)
            return result

    def initial_publish(self, version: int) -> PublishResult:
        return self.publish(self._export_version(int(version)))

    def prune(self) -> list[int]:
        """Reserve victims before HTTP so a late unleased request cannot race their unload."""
        unloaded: list[int] = []
        errors: list[str] = []
        with self._maintenance_lock:
            with self._registry_lock:
                now = time.monotonic()
                victims = [
                    (v, s)
                    for v, s in self._registry.items()
                    if v < self._published_version
                    and s.leases == 0
                    and s.in_flight == 0
                    and s.zero_since_mono is not None
                    and now - s.zero_since_mono >= self.unload_grace_s
                    and v not in self._exporting
                ]
                self._retiring.update(v for v, _ in victims)
            for version, state in victims:
                try:
                    self._unload_adapter(state)
                except Exception as exc:
                    errors.append(f"unload version {version}: {exc}")
                else:
                    with self._registry_lock:
                        state.loaded = False
                    if self._owns_export_dir:
                        try:
                            shutil.rmtree(state.path, ignore_errors=True)
                            if state.path.exists():
                                raise OSError(f"export directory still exists: {state.path}")
                        except Exception as exc:
                            errors.append(f"remove version {version}: {exc}")
                            continue
                    with self._registry_lock:
                        del self._registry[version]
                    unloaded.append(version)
                finally:
                    with self._registry_lock:
                        self._retiring.discard(version)
            self.last_cleanup_errors = errors
        return unloaded

    async def _stop_cleanup(self) -> None:
        with self._registry_lock:
            self._closed = True
        tasks = list(self._cleanup_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class AsyncVllmServerPolicy(LeasedAdapterRegistry, VllmServerPolicy):
    """Keep an episode's adapter alive while learner and sampler progress independently.

    Producers lease/release and generate on the event-loop thread. The learner exports and
    calls its non-blocking sink while holding the backend model lock. Publisher workers load,
    commit and prune; release schedules cleanup off the loop after the grace period. Registry
    critical sections contain only state operations. A separate maintenance lock serializes
    HTTP/unload/delete sequences and is never acquired inside the registry lock. Startup is
    before producers; close is after producers and learner/publisher workers have stopped.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        run_gen: str,
        export_adapter: Callable[[int], Path | str | AdapterSnapshot] | None = None,
        weight_version: int = 0,
        logprobs_mode: str = "processed_logprobs",
        timeout_s: float = 3600.0,
        admin_timeout_s: float = 60.0,
        max_connections: int = 128,
        api_key_var: str | None = None,
        name: str = "vllm-server",
        lora_name_prefix: str = "policy",
        publish_sink: Callable[[PublishRequest], None] | None = None,
        unload_grace_s: float = 2.0,
        adapter_export_root: str | Path | None = None,
        owns_export_dir: bool = True,
    ) -> None:
        VllmServerPolicy.__init__(
            self,
            base_url,
            model,
            export_adapter=export_adapter,
            weight_version=weight_version,
            logprobs_mode=logprobs_mode,
            timeout_s=timeout_s,
            admin_timeout_s=admin_timeout_s,
            max_connections=max_connections,
            api_key_var=api_key_var,
            name=name,
            lora_name_prefix=lora_name_prefix,
        )
        self._init_registry(
            run_gen=run_gen,
            weight_version=weight_version,
            prefix=lora_name_prefix,
            export_adapter=export_adapter,
            publish_sink=publish_sink,
            unload_grace_s=unload_grace_s,
            adapter_export_root=(
                Path(adapter_export_root) if adapter_export_root is not None else None
            ),
            owns_export_dir=owns_export_dir,
        )

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        version, model = self._begin_generate(request)
        try:
            if not request.prompt_ids:
                raise ValueError("GenerateRequest.prompt_ids must not be empty")
            body = self.build_body(request, model=model)
            response = await self._client().post(
                "/v1/completions", json=body, headers=self._headers
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"vLLM /v1/completions returned {response.status_code}: {response.text[:500]}"
                )
            return self.parse_response(response.json(), request, version, served_model=model)
        finally:
            self._end_generate(version)

    def _unload_adapter(self, state: _VersionState) -> None:
        self._admin_post("/v1/unload_lora_adapter", {"lora_name": state.name}, ok_404=True)

    def _load_adapter(self, state: _VersionState) -> None:
        self._admin_post(
            "/v1/load_lora_adapter", {"lora_name": state.name, "lora_path": str(state.path)}
        )

    def initial_publish(self, version: int) -> PublishResult:
        """Remove previous processes' adapters before any producer can hold a lease."""
        with self._maintenance_lock:
            self.cleanup_stale_adapters()
            return super().initial_publish(version)

    def cleanup_stale_adapters(self) -> set[str]:
        """Unload adapters from previous runs and reconcile local loaded state."""
        with self._maintenance_lock:
            payload = self.check_server()
            unloaded_names: set[str] = set()
            try:
                for model in payload.get("data", []):
                    model_id = model.get("id", "")
                    if model_id.startswith(f"{self._prefix}-") and model_id != self._base_model:
                        self._admin_post(
                            "/v1/unload_lora_adapter", {"lora_name": model_id}, ok_404=True
                        )
                        unloaded_names.add(model_id)
            finally:
                with self._registry_lock:
                    for state in self._registry.values():
                        if state.name in unloaded_names:
                            state.loaded = False
        return unloaded_names

    async def close(self) -> None:
        """Best-effort shutdown must not mask an earlier rollout or training failure."""

        async def finish() -> None:
            await self._stop_cleanup()
            await asyncio.to_thread(self._requests_drained.wait)
            if self._async_client is not None:
                try:
                    await self._async_client.aclose()
                except Exception:
                    pass
                self._async_client = None

            def unload_run() -> None:
                with self._maintenance_lock:
                    with self._registry_lock:
                        names = {state.name for state in self._registry.values()}
                    try:
                        payload = self.check_server()
                        names.update(
                            model["id"]
                            for model in payload.get("data", [])
                            if model.get("id", "").startswith(f"{self._prefix}-{self.run_gen}-v")
                            and model["id"]
                            .removeprefix(f"{self._prefix}-{self.run_gen}-v")
                            .isdigit()
                            and model["id"] != self._base_model
                        )
                    except Exception:
                        pass
                    for model in sorted(names):
                        try:
                            self._admin_post(
                                "/v1/unload_lora_adapter", {"lora_name": model}, ok_404=True
                            )
                        except Exception:
                            pass

            try:
                await asyncio.to_thread(unload_run)
            except Exception:
                pass

        worker = asyncio.create_task(finish())
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            await worker
            raise


__all__ = ["AsyncVllmServerPolicy", "LeasedAdapterRegistry"]
