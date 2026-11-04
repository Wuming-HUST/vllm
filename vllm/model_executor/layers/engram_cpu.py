# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU compression, hash and retrieval, outside device graph capture.

Callers own request history and output buffers (including pinned buffers).
These operators do not perform transfers, collectives, or speculative commits.
Importing this module does not enable a serving backend.
"""

import contextlib
import importlib
import threading

import torch


def load_ops() -> None:
    """Load the wheel's host extension without a runtime compiler."""
    if hasattr(torch.ops._engram_cpu, "gather_out"):
        return
    try:
        importlib.import_module("vllm.engram_cpu_C")
    except ImportError as exc:
        raise RuntimeError(
            "Engram CPU operators are unavailable; build the engram_cpu_C "
            "CMake target with the installed PyTorch version."
        ) from exc


def capabilities() -> dict[str, bool | int]:
    """Report compiled and usable ISAs; vector length is for the calling thread."""
    load_ops()
    return {
        "neon": torch.ops._engram_cpu.has_neon(),
        "sve2": torch.ops._engram_cpu.has_sve2(),
        "sve_vector_bytes": torch.ops._engram_cpu.sve_vector_bytes(),
    }


def sources_out(
    ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    lookback: torch.Tensor,
    dead_mask: torch.Tensor,
    lookback_dead_mask: torch.Tensor,
    token_map: torch.Tensor,
    pad_id: int,
    out: torch.Tensor,
    cache: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    cache_block_size: int = 0,
) -> None:
    """Write INT32 [T, K] sources from ragged requests and raw lookback IDs.

    Index inputs may be INT32 or INT64. Lookback is [R, K-1], newest first.
    An unknown lookback (-1) at a nonnegative position contributes pad_id;
    a pre-sequence position or an image mask contributes DEAD (-1). The hash
    operator stops looking further back after the first DEAD.
    """
    load_ops()
    torch.ops._engram_cpu.sources_out(
        ids,
        query_start_loc,
        positions,
        lookback,
        dead_mask,
        lookback_dead_mask,
        token_map,
        pad_id,
        out,
        cache,
        slot_mapping,
        block_table,
        cache_block_size,
    )


def hash_out(
    sources: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    out: torch.Tensor,
    use_sve2: bool = False,
) -> None:
    """Write global INT32 [T, layers, heads] IDs using uint64 hash arithmetic."""
    load_ops()
    torch.ops._engram_cpu.hash_out(
        sources,
        multipliers,
        primes,
        offsets,
        pad_id,
        out,
        use_sve2,
    )


def gather_out(
    codes: torch.Tensor,
    scales: torch.Tensor,
    ids: torch.Tensor,
    head_start: int,
    valid_heads: int,
    vocab_start: int,
    block_size: int,
    out: torch.Tensor,
    backend: str = "auto",
    prefetch_distance: int = 0,
    legacy_e8m0: bool = False,
) -> None:
    """Gather a head shard into BF16 [T, padded_heads, dim] without temporaries.

    Storage: E4M3 + uint8 E8M0 scales, INT8 + FP32 scales, or BF16 + empty
    scales. Invalid IDs and padded heads write zeros. Every output buffer must
    be contiguous, disjoint from inputs and exclusively owned until return.
    Backend is auto (NEON when available), scalar, neon or sve2. SVE2 gather
    and row prefetch are explicit candidates until target measurements exist.
    """
    load_ops()
    torch.ops._engram_cpu.gather_out(
        codes,
        scales,
        ids,
        head_start,
        valid_heads,
        vocab_start,
        block_size,
        out,
        backend,
        prefetch_distance,
        legacy_e8m0,
    )


class CPUEmbeddingLookup:
    """Graph-external device-ID -> CPU retrieval -> device-row bridge.

    Each embedding owns one high-water staging allocation. A native host
    callback orders CPU retrieval after D2H and before H2D on the producer
    stream. A completion event protects pinned storage until reuse. Callers
    publish their layer-ready events after H2D and existing collectives.
    """

    def __init__(
        self,
        *,
        backend: str = "auto",
        prefetch_distance: int = 0,
        legacy_e8m0: bool = False,
    ) -> None:
        load_ops()
        features = capabilities()
        if backend not in ("auto", "scalar", "neon", "sve2"):
            raise ValueError(f"unknown CPU gather backend: {backend}")
        if backend in ("neon", "sve2") and not features[backend]:
            raise RuntimeError(f"requested CPU Engram kernel {backend} is unavailable")
        if prefetch_distance < 0:
            raise ValueError("prefetch_distance must be nonnegative")
        self.backend = backend
        self.prefetch_distance = prefetch_distance
        self.legacy_e8m0 = legacy_e8m0
        self._lock = threading.Lock()
        self._ids: torch.Tensor | None = None
        self._rows: torch.Tensor | None = None
        self._layout: tuple[int, int, int, bool] | None = None
        self._copy_done = None
        self._job = None

    def __del__(self):
        # Normal model teardown must not release the last pinned tensor owner
        # on the runtime callback thread. Retire while Python owns the ticket.
        # Device/runtime teardown failures cannot be recovered in a destructor.
        with contextlib.suppress(Exception):
            self.retire()

    def check(self) -> None:
        """Report CPU callback errors after dispatch and before accepting output."""
        if self._job is not None:
            self._job.wait()

    def retire(self) -> None:
        """Wait before overwriting or releasing the pinned copy source."""
        if self._copy_done is not None:
            self._copy_done.synchronize()
            self._copy_done = None
        if self._job is not None:
            job, self._job = self._job, None
            job.wait()

    def lookup(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor | None,
        indices: torch.Tensor,
        out: torch.Tensor,
        *,
        head_start: int,
        valid_heads: int,
        vocab_start: int,
        block_size: int,
    ) -> None:
        if indices.dtype != torch.int32 or indices.ndim != 2:
            raise ValueError("CPU Engram requires INT32 [tokens, global_heads] IDs")
        if (
            out.ndim != 3
            or out.shape[0] != indices.shape[0]
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
            or indices.device != out.device
        ):
            raise ValueError("invalid CPU Engram device input/output layout")
        if codes.device.type != "cpu" or (
            scales is not None and scales.device.type != "cpu"
        ):
            raise ValueError("CPU Engram requires host-resident table storage")
        device_api = getattr(torch, out.device.type, None)
        capturing = getattr(device_api, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError("CPU Engram retrieval cannot run inside device capture")
        tokens = indices.shape[0]
        pinned = out.device.type != "cpu"
        if pinned and not torch.ops._engram_cpu.callback_available(out.device.type):
            raise RuntimeError(
                f"CPU Engram host callbacks unavailable for {out.device.type}"
            )
        layout = (indices.shape[1], out.shape[1], out.shape[2], pinned)
        with self._lock:
            self.retire()
            if (
                self._layout != layout
                or self._ids is None
                or self._ids.shape[0] < tokens
            ):
                self._ids = torch.empty(
                    (tokens, indices.shape[1]),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=pinned,
                )
                self._rows = torch.empty(
                    tuple(out.shape),
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=pinned,
                )
                self._layout = layout
            assert self._ids is not None and self._rows is not None
            ids, rows = self._ids[:tokens], self._rows[:tokens]
            arguments = (
                codes,
                scales if scales is not None else torch.empty(0, device="cpu"),
                ids,
                head_start,
                valid_heads,
                vocab_start,
                block_size,
                rows,
                self.backend,
                self.prefetch_distance,
                self.legacy_e8m0,
            )
            # Even an enqueue failure must leave a lease on any submitted D2H.
            event = device_api.Event() if pinned else None
            try:
                ids.copy_(indices, non_blocking=pinned)
                if pinned:
                    stream = device_api.current_stream(out.device)
                    # NPUStream.stream() flushes the framework submission queue;
                    # fetch the raw handle only after the D2H is submitted.
                    handle = getattr(stream, f"{out.device.type}_stream")
                    self._job = torch.ops._engram_cpu.enqueue_gather_out(
                        *arguments, handle, out.device.type
                    )
                else:
                    gather_out(*arguments)
                out.copy_(rows, non_blocking=pinned)
            finally:
                if event is not None:
                    self._copy_done = event
                    event.record(device_api.current_stream(out.device))


class CPUHashWorkspace:
    """CPU compression/hash with runner-owned history and reusable outputs.

    Metadata is copied only after its producer stream is ready. The optional
    V1 slot cache is owned by NgramHashState and reset on KV storage rebind.
    Results are borrowed until the next call; publish() leases pinned storage
    until its H2D completes. CPU work must never run during device capture.
    """

    def __init__(self, token_map, multipliers, primes, offsets, pad_id):
        load_ops()
        self.token_map = token_map.detach().to("cpu").contiguous()
        self.multipliers = multipliers.detach().to("cpu").contiguous()
        layers = multipliers.shape[0]
        self.primes = primes.detach().to("cpu").reshape(layers, -1).contiguous()
        self.offsets = offsets.detach().to("cpu").reshape(layers, -1).contiguous()
        self.pad_id = pad_id
        self._inputs: dict[str, torch.Tensor] = {}
        self._sources: torch.Tensor | None = None
        self._hashes: torch.Tensor | None = None
        self._copy_done = None

    def _host(self, name, tensor, *, pin=False):
        tensor = tensor.detach()
        previous = self._inputs.get(name)
        if (
            previous is None
            or previous.dtype != tensor.dtype
            or previous.shape != tensor.shape
            or previous.is_pinned() != pin
        ):
            previous = torch.empty(
                tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=pin
            )
            self._inputs[name] = previous
        previous.copy_(tensor, non_blocking=False)
        return previous

    def run(
        self,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask,
        slot_mapping=None,
        block_table=None,
        *,
        cache=None,
        block_size=0,
        use_sve2=False,
    ):
        device_api = getattr(torch, input_ids.device.type, None)
        capturing = getattr(device_api, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError("CPU Engram hash must be refreshed outside capture")
        if self._copy_done is not None:
            self._copy_done.synchronize()
            self._copy_done = None
        pinned = input_ids.device.type != "cpu"
        query = self._host("query", query_start_loc, pin=pinned)
        if query.numel() == 0:
            raise ValueError("query offsets must contain an initial zero")
        requests = query.numel() - 1
        tokens = int(query[-1])
        self.valid_tokens = tokens
        if tokens < 0 or tokens > input_ids.numel():
            raise ValueError("query offsets exceed the input token count")
        if lookback_token_ids.shape[0] < requests:
            raise ValueError("lookback rows do not cover requests")
        args = [
            self._host("ids", input_ids[:tokens], pin=pinned),
            query,
            self._host("positions", positions[:tokens], pin=pinned),
            self._host("lookback", lookback_token_ids[:requests], pin=pinned),
            self._host("dead", dead_mask[:tokens], pin=pinned),
            self._host("lookback_dead", lookback_dead_mask[:requests], pin=pinned),
        ]
        width = self.multipliers.shape[1]
        if self._sources is None or self._sources.shape[0] < input_ids.numel():
            capacity = input_ids.numel()
            self._sources = torch.empty(
                (capacity, width), dtype=torch.int32, device="cpu"
            )
            self._hashes = torch.empty(
                (capacity, *self.primes.shape),
                dtype=torch.int32,
                device="cpu",
                pin_memory=pinned,
            )
        assert self._hashes is not None
        sources = self._sources[:tokens]
        sources_out(
            *args,
            self.token_map,
            self.pad_id,
            sources,
            cache,
            self._host("slots", slot_mapping[:tokens], pin=pinned)
            if cache is not None
            else None,
            self._host("blocks", block_table, pin=pinned)
            if cache is not None
            else None,
            block_size,
        )
        output = self._hashes[: input_ids.numel()]
        output.fill_(-1)
        hash_out(
            sources,
            self.multipliers,
            self.primes,
            self.offsets,
            self.pad_id,
            output[:tokens],
            use_sve2,
        )
        return output

    def publish(self, hashes, device):
        """Publish hashes on the existing stream before its collectives."""
        if device.type == "cpu":
            return hashes.clone()
        result = hashes.to(device, non_blocking=True)
        api = getattr(torch, device.type)
        self._copy_done = api.Event()
        self._copy_done.record(api.current_stream(device))
        return result
