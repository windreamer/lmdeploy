# Copyright (c) OpenMMLab. All rights reserved.
"""TM GenericCopy copy_cache opt-in: selection behavior + numeric parity.

dtype coverage conclusion (see turbomind_cache gate comment): GenericCopy
covers every KV pool dtype the PyTorch engine resolves -- fp16, bf16,
fp8_e4m3fn, uint8 -- except ``torch.float8_e5m2``, which the TurboMind
DLPack binding does not map; that variant stays on triton (and force
raises at build time per spec user story 7).
"""
import pytest
import torch

from lmdeploy.pytorch.backends.cuda.cache import CudaBlockCacheCopy, CudaCacheBackend
from lmdeploy.pytorch.engine.cache_engine.layout import CacheAllocation, CachePool


def _turbomind_available():
    try:
        from lmdeploy.turbomind import is_available
        return bool(is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

TM_NUMICTEST = pytest.mark.skipif(not _turbomind_available(), reason='TurboMind native extension required')


def _make_allocation(shape, entry_axis, dtype, num_logical_blocks, pages_per_block, num_pools=1):
    pools = []
    views = []
    for _ in range(num_pools):
        tensor = torch.zeros(shape, dtype=dtype, device='cuda')
        pools.append(CachePool(tensor=tensor, entry_axis=entry_axis))
        views.append(tensor)
    return CacheAllocation(pools=tuple(pools), tensor_views=tuple(views))


def _fill_and_expect(cache, entry_axis, num_logical_blocks, pages_per_block, src_blocks):
    logical = cache.unflatten(entry_axis, (num_logical_blocks, pages_per_block))
    for block_id in set(src_blocks.tolist()):
        logical.select(entry_axis, block_id).fill_(float(block_id + 1))
    return logical.index_select(entry_axis, src_blocks).clone()


@pytest.mark.parametrize(
    ('shape', 'entry_axis', 'dtype', 'pages_per_block'),
    [
        ((2, 12, 5), 1, torch.uint8, 2),
        ((18, 3, 7), 0, torch.float16, 3),
        ((18, 3, 7), 0, torch.bfloat16, 3),
        ((2, 3, 12, 4), 2, torch.float8_e4m3fn, 2),
        ((2, 3, 12, 4), 2, torch.float32, 2),
    ],
)
@TM_NUMICTEST
def test_tm_copy_matches_reference_across_kv_dtypes(shape, entry_axis, dtype, pages_per_block, monkeypatch):
    """Numeric parity: TM GenericCopy copy == triton copy == expected bytes."""
    monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=copy_cache')
    num_logical_blocks = shape[entry_axis] // pages_per_block
    src_blocks = torch.tensor([0, 2, 0, 1][:num_logical_blocks], dtype=torch.long, device='cuda')
    dst_blocks = torch.tensor([3, 1, 4, 2][:num_logical_blocks], dtype=torch.long, device='cuda')

    allocation_tm = _make_allocation(shape, entry_axis, dtype, num_logical_blocks, pages_per_block)
    allocation_ref = _make_allocation(shape, entry_axis, dtype, num_logical_blocks, pages_per_block)
    for alloc in (allocation_tm, allocation_ref):
        alloc.pools[0].tensor.fill_(99)
    expected = _fill_and_expect(allocation_tm.pools[0].tensor, entry_axis, num_logical_blocks, pages_per_block,
                                src_blocks)

    copy_tm = CudaCacheBackend.build_block_copy(allocation_tm, num_logical_blocks, pages_per_block)
    copy_tm.copy(src_blocks, dst_blocks)

    copy_ref = CudaCacheBackend.build_block_copy(allocation_ref, num_logical_blocks, pages_per_block)
    copy_ref.copy(src_blocks, dst_blocks)

    assert torch.equal(allocation_tm.pools[0].tensor, allocation_ref.pools[0].tensor)
    assert torch.equal(allocation_tm.pools[0].tensor.unflatten(entry_axis, (
        num_logical_blocks,
        pages_per_block,
    )).index_select(entry_axis, dst_blocks), expected)


@TM_NUMICTEST
def test_tm_copy_handles_multiple_pools(monkeypatch):
    """One copy call moves every owning pool of the allocation."""
    monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'force=copy_cache')
    shape, entry_axis, pages_per_block = (6, 4, 3), 0, 2
    num_logical_blocks = shape[entry_axis] // pages_per_block
    allocation = _make_allocation(shape, entry_axis, torch.float16, num_logical_blocks, pages_per_block, num_pools=2)
    src_blocks = torch.tensor([0, 1], dtype=torch.long, device='cuda')
    dst_blocks = torch.tensor([2, 0], dtype=torch.long, device='cuda')
    expected = []
    for pool in allocation.pools:
        logical = pool.tensor.unflatten(entry_axis, (num_logical_blocks, pages_per_block))
        logical.select(entry_axis, 0).fill_(1.0)
        logical.select(entry_axis, 1).fill_(2.0)
        expected.append(logical.index_select(entry_axis, src_blocks).clone())

    copy = CudaCacheBackend.build_block_copy(allocation, num_logical_blocks, pages_per_block)
    copy.copy(src_blocks, dst_blocks)

    for pool, want in zip(allocation.pools, expected):
        logical = pool.tensor.unflatten(entry_axis, (num_logical_blocks, pages_per_block))
        assert torch.equal(logical.index_select(entry_axis, dst_blocks), want)


def test_off_and_auto_keep_triton_copy(monkeypatch):
    """off and default auto both build the existing Triton copy primitive."""
    allocation = _make_allocation((6, 4, 3), 0, torch.float16, 3, 2)
    for env in (None, 'off'):
        if env is None:
            monkeypatch.delenv('LMDEPLOY_TURBOMIND_OPS', raising=False)
        else:
            monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', env)
        copy = CudaCacheBackend.build_block_copy(allocation, 3, 2)
        assert type(copy) is CudaBlockCacheCopy


def test_auto_keeps_triton_for_unsupported_fp8_e5m2(monkeypatch):
    """fp8_e5m2 KV has no TM DLPack mapping: auto keeps triton even when a
    factory-table row existed; the variant is excluded by the gate."""
    monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', 'auto')
    allocation = _make_allocation((6, 4, 3), 0, torch.float8_e5m2, 3, 2)
    copy = CudaCacheBackend.build_block_copy(allocation, 3, 2)
    assert type(copy) is CudaBlockCacheCopy


def test_force_with_unsupported_fp8_e5m2_raises(monkeypatch):
    """force covering copy_cache must fail the feasibility gate at build
    time instead of silently degrading to triton."""
    for env in ('force=copy_cache', 'force'):
        monkeypatch.setenv('LMDEPLOY_TURBOMIND_OPS', env)
        allocation = _make_allocation((6, 4, 3), 0, torch.float8_e5m2, 3, 2)
        with pytest.raises(RuntimeError, match='copy_cache'):
            CudaCacheBackend.build_block_copy(allocation, 3, 2)
