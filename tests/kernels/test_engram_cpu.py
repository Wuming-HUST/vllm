# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contracts; separate from test_engram.py's Triton/CUDA imports.

The oracle uses Python integer arithmetic and Torch decoding, not the native
implementation. No model, scheduler, graph, or serving acceptance is implied.
"""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def load_host_extension():
    library = Path(__file__).resolve().parents[2] / "vllm/engram_cpu_C.abi3.so"
    spec = importlib.util.spec_from_file_location("engram_cpu_C", library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@pytest.fixture(params=["scalar", "auto", "neon", "sve2"])
def gather_backend(request, load_host_extension):
    backend = request.param
    available = {
        "neon": torch.ops._engram_cpu.has_neon(),
        "sve2": torch.ops._engram_cpu.has_sve2(),
    }
    if backend in available and not available[backend]:
        pytest.skip(f"{backend} is unavailable on this build/host")
    return backend


@pytest.fixture(scope="module")
def cpu_api(load_host_extension):
    # The API depends only on Torch. Load it directly to isolate the real
    # native bridge from this checkout's uninstalled serving dependencies.
    path = (
        Path(__file__).resolve().parents[2] / "vllm/model_executor/layers/engram_cpu.py"
    )
    spec = importlib.util.spec_from_file_location("engram_cpu_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("legacy", [False, True])
def test_device_bridge_reuses_staging_without_stale_rows(cpu_api, legacy):
    bridge = cpu_api.CPUEmbeddingLookup(legacy_e8m0=legacy)
    codes = torch.arange(256, dtype=torch.uint8).view(1, 256).repeat(5, 1)
    codes = codes.view(torch.float8_e4m3fn)
    scales = torch.tensor([0, 1, 127, 254, 255], dtype=torch.uint8)[:, None].repeat(
        1, 8
    )
    if legacy:
        sf = (scales.int() << 23).view(torch.float32)
    else:
        sf = scales.view(torch.float8_e8m0fnu).float()
    decoded = (codes.float().view(5, 8, 32) * sf[..., None]).flatten(-2).bfloat16()
    saved = None
    for tokens in (5, 0, 2, 7, 1):
        # Strided input columns model slices of [tokens, layers, heads].
        ids = (torch.arange(tokens * 8, dtype=torch.int32).view(tokens, 8) % 5)[:, ::2]
        out = torch.empty((tokens, 3, 256), dtype=torch.bfloat16)
        bridge.lookup(
            codes,
            scales,
            ids,
            out,
            head_start=1,
            valid_heads=2,
            vocab_start=0,
            block_size=32,
        )
        torch.testing.assert_close(
            out[:, :2], decoded[ids[:, 1:3].long()], rtol=0, atol=0, equal_nan=True
        )
        assert out[:, 2:].count_nonzero() == 0
        if tokens == 5:
            saved = bridge._rows.data_ptr()
        if tokens in (0, 2):
            assert bridge._rows.data_ptr() == saved


def test_device_bridge_serializes_staging_ownership(cpu_api):
    bridge = cpu_api.CPUEmbeddingLookup()
    codes = torch.arange(8 * 32, dtype=torch.float32).view(8, 32).bfloat16()

    def run(index):
        ids = torch.full((31, 2), index % 8, dtype=torch.int32)
        out = torch.empty((31, 2, 32), dtype=torch.bfloat16)
        bridge.lookup(
            codes,
            None,
            ids,
            out,
            head_start=0,
            valid_heads=2,
            vocab_start=0,
            block_size=32,
        )
        torch.testing.assert_close(out, codes[index % 8].expand_as(out), rtol=0, atol=0)

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(run, range(16)))


def test_device_bridge_rejects_invalid_selection_and_id_layout(cpu_api):
    with pytest.raises(ValueError, match="unknown CPU gather backend"):
        cpu_api.CPUEmbeddingLookup(backend="missing")
    bridge = cpu_api.CPUEmbeddingLookup()
    with pytest.raises(ValueError, match="INT32"):
        bridge.lookup(
            torch.ones((1, 32), dtype=torch.bfloat16),
            None,
            torch.zeros((1, 1), dtype=torch.int64),
            torch.empty((1, 1, 32), dtype=torch.bfloat16),
            head_start=0,
            valid_heads=1,
            vocab_start=0,
            block_size=32,
        )


def test_cpu_slot_cache_uses_generated_history_and_rewrites_rejected_slots(cpu_api):
    lut = torch.arange(32, dtype=torch.int32) // 2
    cache = torch.zeros(12, dtype=torch.int32)
    blocks = torch.tensor([[1, 2]], dtype=torch.int32)

    def run(ids, start, known=None, dead=None):
        ids = torch.tensor(ids, dtype=torch.int32)
        out = torch.empty((len(ids), 4), dtype=torch.int32)
        cpu_api.sources_out(
            ids,
            torch.tensor([0, len(ids)], dtype=torch.int32),
            torch.arange(start, start + len(ids)),
            torch.tensor([known or [-1, -1, -1]], dtype=torch.int32),
            torch.tensor(dead or [False] * len(ids)),
            torch.zeros((1, 3), dtype=torch.bool),
            lut,
            0,
            out,
            cache,
            torch.arange(4 + start, 4 + start + len(ids)),
            blocks,
            4,
        )
        return out

    torch.testing.assert_close(
        run([2, 4, 6], 0),
        torch.tensor(
            [[1, -1, -1, -1], [2, 1, -1, -1], [3, 2, 1, -1]], dtype=torch.int32
        ),
    )
    # Generated lookback is unknown to V1's CPU token table; slot history wins.
    assert run([8, 10], 3, known=[-1, 4, -1]).tolist() == [[4, 3, 2, 1], [5, 4, 3, 2]]
    # Rejected draft positions are overwritten by the next scheduled chunk.
    assert run([12], 3).tolist() == [[6, 3, 2, 1]]
    assert run([14], 4).tolist() == [[7, 6, 3, 2]]
    assert run([30], 5, dead=[True]).tolist() == [[-1, 7, 6, 3]]
    assert run([16], 6).tolist() == [[8, -1, 7, 6]]


@pytest.mark.parametrize("slot_cache", [False, True])
def test_cpu_hash_workspace_compresses_hashes_and_clears_capacity_padding(
    cpu_api, slot_cache
):
    lut = torch.arange(32, dtype=torch.int32) // 2
    mult = torch.tensor([[3, 5, 7, 9], [11, 13, 15, 17]])
    primes = torch.tensor([[[19], [23], [29]]] * 2)
    offsets = torch.tensor([[0, 19, 42]] * 2)
    workspace = cpu_api.CPUHashWorkspace(lut, mult, primes, offsets, 0)
    cache = torch.zeros(16, dtype=torch.int32) if slot_cache else None
    previous_pointer = None
    for step, count in enumerate((3, 1, 0, 2)):
        # Last two raw IDs are capacity padding and cannot enter history.
        raw = torch.tensor(list(range(2 + 2 * step, 2 + 2 * step + count)) + [99, 99])
        positions = torch.arange(count + 2)
        query = torch.tensor([0, count], dtype=torch.int32)
        old = torch.full((1, 3), -1, dtype=torch.int32)
        result = workspace.run(
            raw,
            positions,
            query,
            torch.zeros(count + 2, dtype=torch.bool),
            old,
            torch.zeros_like(old, dtype=torch.bool),
            torch.arange(count) if slot_cache else None,
            torch.tensor([[0, 1, 2, 3]]) if slot_cache else None,
            cache=cache,
            block_size=4,
        )
        assert workspace.valid_tokens == count
        assert result[count:].eq(-1).all()
        if previous_pointer is not None:
            assert result.data_ptr() == previous_pointer
        previous_pointer = result.data_ptr()
        expected = torch.empty((count, 2, 3), dtype=torch.int32)
        for token in range(count):
            for layer in range(2):
                rolling = 0
                for shift in range(4):
                    value = int(lut[raw[token - shift]]) if shift <= token else 0
                    rolling ^= value * int(mult[layer, shift])
                    if shift:
                        col = shift - 1
                        expected[token, layer, col] = rolling % int(
                            primes[layer, col, 0]
                        ) + int(offsets[layer, col])
        torch.testing.assert_close(result[:count], expected, rtol=0, atol=0)
        published = workspace.publish(result, torch.device("cpu"))
        result.fill_(-1)
        torch.testing.assert_close(published[:count], expected, rtol=0, atol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_native_callback_ticket_reports_completion_and_errors(fail):
    codes = torch.arange(3 * 32, dtype=torch.float32).view(3, 32).bfloat16()
    ids = torch.tensor([[2], [0], [-1]], dtype=torch.int32)
    out = torch.full((3, 1, 32), 99, dtype=torch.bfloat16)
    ticket = torch.ops._engram_cpu.enqueue_gather_out(
        codes,
        torch.empty(0),
        ids,
        0,
        1,
        0,
        32,
        out,
        "bad" if fail else "auto",
        0,
        False,
        0,
        "cpu",
    )
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="unknown CPU gather backend"):
                ticket.wait()
            assert out.count_nonzero() == 0
        else:
            ticket.wait()
            torch.testing.assert_close(out[:2, 0], codes[[2, 0]], rtol=0, atol=0)
            assert out[2].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA streams")
def test_cuda_callback_orders_copies_and_refreshes_full_graph(cpu_api):
    bridge = cpu_api.CPUEmbeddingLookup()
    codes = torch.arange(7 * 32, dtype=torch.float32).view(7, 32).bfloat16()
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    ids = torch.zeros((5, 1), dtype=torch.int32, device="cuda")
    rows = torch.zeros((5, 1, 32), dtype=torch.bfloat16, device="cuda")
    result = torch.empty_like(rows)
    ready = torch.cuda.Event(external=True)
    consumer.wait_stream(torch.cuda.current_stream())
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        bridge.lookup(
            codes,
            None,
            ids,
            rows,
            head_start=0,
            valid_heads=1,
            vocab_start=0,
            block_size=32,
        )
        ready.record(producer)
    bridge.check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=consumer):
        ready.wait(consumer)
        result.copy_(rows)
    for step in (1, 5, 2, 6):
        producer.wait_stream(consumer)
        with torch.cuda.stream(producer):
            ids.fill_(step)
            bridge.lookup(
                codes,
                None,
                ids,
                rows,
                head_start=0,
                valid_heads=1,
                vocab_start=0,
                block_size=32,
            )
            ready.record(producer)
        with torch.cuda.stream(consumer):
            graph.replay()
        bridge.check()
        consumer.synchronize()
        torch.testing.assert_close(
            result.cpu(), codes[step].expand_as(result), rtol=0, atol=0
        )
    bridge.retire()


@pytest.mark.parametrize("host_ids", [False, True])
def test_npu_callback_refreshes_external_event_graph(cpu_api, host_ids):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    torch.npu.set_device(0)
    bridge = cpu_api.CPUEmbeddingLookup()
    codes = torch.arange(7 * 32, dtype=torch.float32).view(7, 32).bfloat16()
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    ids = torch.zeros((5, 1), dtype=torch.int32, device="cpu" if host_ids else "npu")
    rows = torch.zeros((5, 1, 32), dtype=torch.bfloat16, device="npu")
    result = torch.empty_like(rows)
    ready = torch.npu.ExternalEvent()
    producer.wait_stream(torch.npu.current_stream())
    consumer.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(producer):
        bridge.lookup(
            codes,
            None,
            ids,
            rows,
            head_start=0,
            valid_heads=1,
            vocab_start=0,
            block_size=32,
        )
        ready.record(producer)
    bridge.check()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=consumer):
        ready.wait(consumer)
        ready.reset(consumer)
        result.copy_(rows)
    for step in (1, 5, 2, 6):
        producer.wait_stream(consumer)
        with torch.npu.stream(producer):
            ids.fill_(step)
            bridge.lookup(
                codes,
                None,
                ids,
                rows,
                head_start=0,
                valid_heads=1,
                vocab_start=0,
                block_size=32,
            )
            if host_ids:
                ids.fill_((step + 1) % 7)
            ready.record(producer)
        with torch.npu.stream(consumer):
            graph.replay()
        bridge.check()
        consumer.synchronize()
        torch.testing.assert_close(
            result.cpu(), codes[step].expand_as(result), rtol=0, atol=0
        )
    bridge.retire()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_ragged_sources_preserve_boundaries_and_dead_history(dtype):
    ids = torch.tensor([4, 5, 6, 7, 8, 9], dtype=dtype)
    query = torch.tensor([0, 0, 3, 4, 6], dtype=dtype)
    positions = torch.tensor([0, 1, 2, 8, 2, 3], dtype=dtype)
    history = torch.tensor(
        [[-1, -1, -1], [-1, -1, -1], [-1, 3, 2], [1, 2, -1]], dtype=dtype
    )
    dead = torch.tensor([False, True, False, False, False, True])
    old_dead = torch.zeros_like(history, dtype=torch.bool)
    old_dead[3, 1] = True
    lut = torch.arange(16, dtype=torch.int32) // 2
    out = torch.full((6, 4), 999, dtype=torch.int32)
    address = out.data_ptr()
    torch.ops._engram_cpu.sources_out(
        ids, query, positions, history, dead, old_dead, lut, 0, out
    )
    # Missing history at position 7 contributes pad; older known IDs survive.
    expected = [
        [2, -1, -1, -1],
        [-1, 2, -1, -1],
        [3, -1, 2, -1],
        [3, 0, 1, 1],
        [4, 0, -1, -1],
        [-1, 4, 0, -1],
    ]
    assert out.tolist() == expected
    assert out.data_ptr() == address


def test_chunked_sources_match_whole_sequence_after_request_reorder():
    sequences = [[2, 4, 8, 1, 5, 9], [3, 7, 6, 4]]
    lut = torch.arange(16, dtype=torch.int32) // 2

    def run(rows, starts):
        ids, positions, offsets, history = [], [], [0], []
        for row, start in zip(rows, starts):
            ids.extend(row[start:])
            positions.extend(range(start, len(row)))
            offsets.append(len(ids))
            history.append([row[start - s] if start >= s else -1 for s in range(1, 4)])
        raw = torch.tensor(ids, dtype=torch.int32)
        old = torch.tensor(history, dtype=torch.int32)
        out = torch.empty((len(ids), 4), dtype=torch.int32)
        torch.ops._engram_cpu.sources_out(
            raw,
            torch.tensor(offsets, dtype=torch.int32),
            torch.tensor(positions, dtype=torch.int32),
            old,
            raw == 8,
            old == 8,
            lut,
            0,
            out,
        )
        return out

    whole = run(sequences, [0, 0])
    # B is now first in the batch; neither row index nor previous call owns state.
    chunks = run(sequences[::-1], [2, 3])
    torch.testing.assert_close(chunks[:2], whole[8:10])
    torch.testing.assert_close(chunks[2:], whole[3:6])


@pytest.mark.parametrize("width,heads", [(2, 1), (4, 8), (5, 3)])
def test_hash_matches_unsigned_python_oracle(width, heads):
    generator = torch.Generator().manual_seed(73)
    source = torch.randint(
        -1, 99092, (137, width), generator=generator, dtype=torch.int32
    )
    source[0].fill_(-1)
    source[1, 1] = -1
    # Negative int64 bit patterns exercise uint64 wraparound and high products.
    mult = torch.randint(-(1 << 62), (1 << 62), (2, width), generator=generator)
    primes = torch.tensor(
        [[2, 31, 65537, 16000057][h % 4] for h in range((width - 1) * heads)]
    ).repeat(2, 1)
    offsets = (primes.cumsum(1) - primes).contiguous()
    out = torch.empty((len(source), 2, primes.size(1)), dtype=torch.int32)
    torch.ops._engram_cpu.hash_out(source, mult, primes, offsets, 2, out)
    if torch.ops._engram_cpu.has_sve2():
        sve = torch.empty_like(out)
        torch.ops._engram_cpu.hash_out(source, mult, primes, offsets, 2, sve, True)
        torch.testing.assert_close(sve, out, rtol=0, atol=0)
    else:
        with pytest.raises(RuntimeError, match="SVE2 hash is unavailable"):
            torch.ops._engram_cpu.hash_out(source, mult, primes, offsets, 2, out, True)
    expected = torch.empty_like(out)
    for t, row in enumerate(source.tolist()):
        for layer in range(2):
            rolling, blocked = 0, False
            for shift, value in enumerate(row):
                blocked |= value == -1
                rolling ^= ((2 if blocked else value) * int(mult[layer, shift])) & (
                    (1 << 64) - 1
                )
                if shift:
                    for head in range(heads):
                        col = (shift - 1) * heads + head
                        expected[t, layer, col] = rolling % int(
                            primes[layer, col]
                        ) + int(offsets[layer, col])
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("fmt", ["mxfp8", "int8", "bf16"])
def test_all_head_shards_reconstruct_reference_with_tail_padding(
    tp, fmt, gather_backend
):
    generator = torch.Generator().manual_seed(19)
    heads, rows_per_head, dim, block = 24, 7, 256, 32
    rows = heads * rows_per_head
    values = torch.randn((rows, dim), generator=generator)
    if fmt == "mxfp8":
        codes = values.to(torch.float8_e4m3fn)
        scales = torch.randint(
            124, 130, (rows, dim // block), generator=generator, dtype=torch.uint8
        )
        decoded_scales = torch.ldexp(
            torch.ones_like(scales, dtype=torch.float32), scales.int() - 127
        )
    elif fmt == "int8":
        codes = (values * 16).to(torch.int8)
        scales = torch.rand((rows, dim // block), generator=generator)
        decoded_scales = scales
    else:
        codes, scales = values.bfloat16(), torch.empty(0)
    ids = (
        torch.randint(rows_per_head, (5, heads), generator=generator)
        + torch.arange(heads) * rows_per_head
    ).int()
    ids[0, 0] = -1
    ids[1, -1] = rows + 1
    safe = ids.clamp(0, rows - 1).long()
    decoded = codes.float()
    if fmt != "bf16":
        decoded = (
            (decoded.unflatten(-1, (-1, block)) * decoded_scales.unsqueeze(-1))
            .flatten(-2)
            .bfloat16()
        )
    expected = (
        decoded[safe]
        .bfloat16()
        .masked_fill(((ids < 0) | (ids >= rows)).unsqueeze(-1), 0)
    )
    shards = []
    for rank in range(tp):
        count, start = heads // tp, rank * (heads // tp)
        begin, end = start * rows_per_head, (start + count) * rows_per_head
        out = torch.full((5, count + 1, dim), float("nan"), dtype=torch.bfloat16)
        ptr = out.data_ptr()
        torch.ops._engram_cpu.gather_out(
            codes[begin:end].contiguous(),
            scales[begin:end].contiguous() if fmt != "bf16" else scales,
            ids,
            start,
            count,
            begin,
            block,
            out,
            gather_backend,
            2,
        )
        assert out.data_ptr() == ptr
        assert torch.count_nonzero(out[:, count:]) == 0
        shards.append(out[:, :count])
    torch.testing.assert_close(torch.cat(shards, dim=1), expected, rtol=0, atol=0)


def test_mxfp8_all_codes_and_extreme_scales(gather_backend):
    codes = torch.arange(256, dtype=torch.uint8).reshape(1, 256).repeat(5, 1)
    codes = codes.view(torch.float8_e4m3fn)
    scales = torch.tensor([0, 1, 127, 254, 255], dtype=torch.uint8)[:, None].repeat(
        1, 8
    )
    ids = torch.arange(5, dtype=torch.int32)[:, None]
    out = torch.empty((5, 1, 256), dtype=torch.bfloat16)
    torch.ops._engram_cpu.gather_out(
        codes, scales, ids, 0, 1, 0, 32, out, gather_backend
    )
    sf = scales.view(torch.float8_e8m0fnu).float()
    expected = (codes.float().unflatten(-1, (-1, 32)) * sf[..., None]).flatten(-2)
    torch.testing.assert_close(
        out[:, 0], expected.bfloat16(), rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.parametrize("block", [1, 3, 7, 8, 13, 32, 65])
@pytest.mark.parametrize("fmt", ["mxfp8", "int8"])
def test_vector_tails_rounding_and_output_guards(block, fmt, gather_backend):
    generator = torch.Generator().manual_seed(27)
    dim = 2 * block
    raw = torch.randint(0, 256, (11, dim), generator=generator, dtype=torch.uint8)
    if fmt == "mxfp8":
        codes = raw.view(torch.float8_e4m3fn)
        scales = torch.randint(
            120, 134, (11, 2), generator=generator, dtype=torch.uint8
        )
        sf = scales.view(torch.float8_e8m0fnu).float()
    else:
        codes = raw.view(torch.int8)
        scales = torch.rand((11, 2), generator=generator)
        sf = scales
    ids = torch.tensor([[8, 4], [-1, 10]], dtype=torch.int32)
    storage = torch.full((2 * 3 * dim + 2,), 123.0, dtype=torch.bfloat16)
    out = storage[1:-1].view(2, 3, dim)
    torch.ops._engram_cpu.gather_out(
        codes, scales, ids, 0, 2, 0, block, out, gather_backend, 3
    )
    decoded = (codes.float().unflatten(-1, (2, block)) * sf[..., None]).flatten(-2)
    expected = decoded[ids.clamp_min(0).long()].bfloat16()
    expected.masked_fill_((ids < 0)[..., None], 0)
    torch.testing.assert_close(out[:, :2], expected, rtol=0, atol=0, equal_nan=True)
    assert out[:, 2:].count_nonzero() == 0
    assert storage[0] == 123 and storage[-1] == 123


def test_backend_capability_guards():
    codes = torch.ones((1, 8), dtype=torch.int8)
    scales = torch.ones((1, 1))
    ids = torch.zeros((1, 1), dtype=torch.int32)
    out = torch.empty((1, 1, 8), dtype=torch.bfloat16)
    for backend, available in (
        ("neon", torch.ops._engram_cpu.has_neon()),
        ("sve2", torch.ops._engram_cpu.has_sve2()),
    ):
        if not available:
            with pytest.raises(RuntimeError, match="is unavailable"):
                torch.ops._engram_cpu.gather_out(
                    codes, scales, ids, 0, 1, 0, 8, out, backend
                )
    with pytest.raises(RuntimeError, match="unknown CPU gather backend"):
        torch.ops._engram_cpu.gather_out(codes, scales, ids, 0, 1, 0, 8, out, "sve3")
    with pytest.raises(RuntimeError, match="nonnegative"):
        torch.ops._engram_cpu.gather_out(
            codes, scales, ids, 0, 1, 0, 8, out, "auto", -1
        )


def test_int8_bf16_rounding_ties_nan_and_signed_zero(gather_backend):
    block = 13
    scales = torch.tensor(
        [
            [
                0.0,
                -0.0,
                float("inf"),
                -float("inf"),
                float("nan"),
                1 + 2**-8,
                1 + 3 * 2**-8,
                2**-127,
            ]
        ]
    )
    codes = torch.ones((1, 8 * block), dtype=torch.int8)
    codes[:, 1::3] = -1
    codes[:, 2::3] = 0
    out = torch.empty((1, 1, codes.size(1)), dtype=torch.bfloat16)
    torch.ops._engram_cpu.gather_out(
        codes,
        scales,
        torch.zeros((1, 1), dtype=torch.int32),
        0,
        1,
        0,
        block,
        out,
        gather_backend,
    )
    expected = (codes.float().view(1, 8, block) * scales[..., None]).bfloat16()
    # Check the sign of zero as well as ties and canonical NaN bits.
    assert torch.equal(
        out.view(torch.int16).flatten(), expected.view(torch.int16).flatten()
    )


def test_empty_batch_and_invalid_memory_contracts():
    codes = torch.empty((0, 32), dtype=torch.int8)
    scales = torch.empty((0, 1))
    ids = torch.empty((0, 24), dtype=torch.int32)
    torch.ops._engram_cpu.gather_out(
        codes, scales, ids, 0, 3, 0, 32, torch.empty((0, 3, 32), dtype=torch.bfloat16)
    )
    # Zero-row shards can still have token work (invalid IDs must be zero).
    out = torch.ones((2, 1, 32), dtype=torch.bfloat16)
    torch.ops._engram_cpu.gather_out(
        codes, scales, torch.zeros((2, 1), dtype=torch.int32), 0, 1, 0, 32, out
    )
    assert out.count_nonzero() == 0
    with pytest.raises(RuntimeError, match="invalid shard range"):
        torch.ops._engram_cpu.gather_out(
            codes,
            scales,
            ids,
            24,
            1,
            0,
            32,
            torch.empty((0, 1, 32), dtype=torch.bfloat16),
        )
    table = torch.ones((1, 32), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="single memory location|overlap"):
        torch.ops._engram_cpu.gather_out(
            table,
            torch.empty(0),
            torch.zeros((1, 1), dtype=torch.int32),
            0,
            1,
            0,
            32,
            table.view(1, 1, 32),
        )


def test_sources_empty_requests_and_bad_metadata():
    def run(ids, query, positions, out):
        torch.ops._engram_cpu.sources_out(
            torch.tensor(ids, dtype=torch.int32),
            torch.tensor(query, dtype=torch.int32),
            torch.tensor(positions, dtype=torch.int32),
            torch.full((2, 3), -1, dtype=torch.int32),
            torch.zeros(len(ids), dtype=torch.bool),
            torch.zeros((2, 3), dtype=torch.bool),
            torch.arange(10, dtype=torch.int32),
            0,
            out,
        )

    run([], [0, 0, 0], [], torch.empty((0, 4), dtype=torch.int32))
    out = torch.empty((2, 4), dtype=torch.int32)
    with pytest.raises(RuntimeError, match="monotonic"):
        run([1, 2], [0, 3, 2], [0, 1], out)
    with pytest.raises(RuntimeError, match="positions"):
        run([1, 2], [0, 2, 2], [0, 2], out)
    with pytest.raises(RuntimeError, match="outside token_map"):
        run([1, 10], [0, 2, 2], [0, 1], out)
