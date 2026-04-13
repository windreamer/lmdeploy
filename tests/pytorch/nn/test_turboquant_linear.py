"""Tests for TurboQuantLinear module.

This module contains unit tests for TurboQuantLinear, verifying:
- Creation and parameter registration
- Forward pass correctness (vs bf16 baseline)
- Weight-only inference
- Codebook and boundaries parameter registration
"""

import pytest
import torch

from lmdeploy.pytorch.nn.linear.turboquant import MergedTurboQuantLinear, TurboQuantLinear


class TestTurboQuantLinearCreation:
    """Test TurboQuantLinear creation and parameter registration."""

    @pytest.fixture
    def in_features(self):
        yield 128

    @pytest.fixture
    def out_features(self):
        yield 256

    def test_creation_basic(self, in_features, out_features):
        """Test basic creation without bias."""
        layer = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

        assert layer.in_features == in_features
        assert layer.out_features == out_features
        assert layer.bias is None
        assert layer.weight is not None
        assert layer.weight_norms is not None
        # Check weight shape: (out_features, in_features // 2) for 4-bit
        assert layer.weight.shape == (out_features, in_features // 2)
        # Check norms shape: (out_features, n_groups)
        n_groups = (in_features + 128 - 1) // 128
        assert layer.weight_norms.shape == (out_features, n_groups)

    def test_creation_with_bias(self, in_features, out_features):
        """Test creation with bias."""
        layer = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
        )

        assert layer.bias is not None
        assert layer.bias.shape == (out_features,)

    def test_creation_with_colwise(self, in_features, out_features):
        """Test creation with colwise=True."""
        layer = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            colwise=True,
        )

        assert layer.colwise is True


class TestTurboQuantLinearForward:
    """Test TurboQuantLinear forward pass correctness."""

    @pytest.fixture
    def in_features(self):
        yield 128

    @pytest.fixture
    def out_features(self):
        yield 256

    @pytest.fixture
    def batch_size(self):
        yield 4

    @pytest.fixture
    def seq_len(self):
        yield 8

    def test_forward_correctness(self, in_features, out_features, batch_size, seq_len):
        """Test forward pass correctness vs bf16 baseline.

        TurboQuant is a lossy quantization method with random rotation. MSE is typically around 1-2 due to the
        quantization error.
        """
        torch.manual_seed(42)

        # Create TurboQuantLinear on CUDA and load random weight
        tq_layer = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        ).cuda()

        # Create random bf16 weight and load it (will be quantized on-the-fly)
        W_bf16 = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        tq_layer._weight_loader_with_quant(tq_layer.weight, W_bf16)

        # Create input
        x = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16).cuda()
        tq_layer = tq_layer.cuda()

        # Forward pass
        out_tq = tq_layer(x)

        # Reference: bf16 matmul
        out_ref = torch.matmul(x, W_bf16.cuda().t())

        # Check output shape
        assert out_tq.shape == out_ref.shape

        # Check MSE (TurboQuant has inherent quantization error)
        mse = torch.mean((out_tq.float() - out_ref.float()) ** 2)
        print(f'  Forward MSE: {mse.item():.6f}')
        # TurboQuant typically has MSE around 1-2 due to lossy quantization
        assert mse < 5.0, f'Forward MSE too high: {mse.item()}'

    def test_forward_with_bias(self, in_features, out_features, batch_size, seq_len):
        """Test forward pass with bias."""
        torch.manual_seed(42)

        # Create TurboQuantLinear with bias on CUDA
        tq_layer = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
        ).cuda()

        # Load random weight
        W_bf16 = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        tq_layer._weight_loader_with_quant(tq_layer.weight, W_bf16)

        # Create input
        x = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16).cuda()
        tq_layer = tq_layer.cuda()

        # Forward pass
        out_tq = tq_layer(x)

        # Reference: bf16 matmul + bias
        out_ref = torch.matmul(x, W_bf16.cuda().t()) + tq_layer.bias

        # Check MSE (TurboQuant has inherent quantization error)
        mse = torch.mean((out_tq.float() - out_ref.float()) ** 2)
        print(f'  Forward with bias MSE: {mse.item():.6f}')
        assert mse < 5.0


class TestTurboQuantLinearWeightOnly:
    """Test weight-only inference (batch inference with pre-quantized
    weights)."""

    @pytest.fixture
    def in_features(self):
        yield 128

    @pytest.fixture
    def out_features(self):
        yield 256

    @pytest.fixture
    def batch_size(self):
        yield 8

    def test_weight_only_inference(self, in_features, out_features, batch_size):
        """Test weight-only inference with pre-quantized weights."""
        torch.manual_seed(42)

        # Create TurboQuantLinear on CUDA
        tq_layer = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        ).cuda()

        # Load and quantize weight
        W_bf16 = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        tq_layer._weight_loader_with_quant(tq_layer.weight, W_bf16)

        # Create input (batch, in_features)
        x = torch.randn(batch_size, in_features, dtype=torch.bfloat16).cuda()
        tq_layer = tq_layer.cuda()

        # Forward pass
        out = tq_layer(x)

        # Verify output shape
        assert out.shape == (batch_size, out_features)

        # Verify output is not all zeros
        assert not torch.all(out == 0)

    def test_get_dequantized_weight(self, in_features, out_features):
        """Test get_dequantized_weight method."""
        torch.manual_seed(42)

        # Create TurboQuantLinear on CUDA
        tq_layer = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        ).cuda()

        # Load weight
        W_bf16 = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        tq_layer._weight_loader_with_quant(tq_layer.weight, W_bf16)

        # Get dequantized weight
        W_dequant = tq_layer.get_dequantized_weight()

        # Check shape
        assert W_dequant.shape == (out_features, in_features)

        # Check dtype
        assert W_dequant.dtype == torch.bfloat16

        # Check reconstruction error - use absolute error for small values
        W_ref = W_bf16.to(torch.bfloat16).cuda()
        # For values > 0.1, check relative error; for small values, check absolute error
        mask = torch.abs(W_ref) > 0.1
        rel_error = torch.mean(torch.abs(W_dequant[mask] - W_ref[mask]) / (torch.abs(W_ref[mask]) + 1e-8))
        abs_error = torch.mean(torch.abs(W_dequant - W_ref))
        print(f'  Weight reconstruction relative error (for |W|>0.1): {rel_error.item():.6f}')
        print(f'  Weight reconstruction absolute error: {abs_error.item():.6f}')
        # Allow higher relative error for 4-bit quantization, but absolute should be small
        assert rel_error < 1.0 and abs_error < 0.2


class TestTurboQuantLinearParameterRegistration:
    """Test codebook and boundaries parameter registration."""

    @pytest.fixture
    def in_features(self):
        yield 128

    @pytest.fixture
    def out_features(self):
        yield 256

    def test_codebook_boundaries_as_buffers(self, in_features, out_features):
        """Test that codebook and boundaries are registered as buffers."""
        layer = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

        # Check that codebook and boundaries are accessible via properties
        assert hasattr(layer, 'codebook')
        assert hasattr(layer, 'boundaries')
        assert isinstance(layer.codebook, torch.Tensor)
        assert isinstance(layer.boundaries, torch.Tensor)

        # Check that they are registered as buffers (not parameters)
        assert '_codebook' in dict(layer.named_buffers())
        assert '_boundaries' in dict(layer.named_buffers())
        assert 'codebook' not in dict(layer.named_parameters())
        assert 'boundaries' not in dict(layer.named_parameters())

        # Check shapes
        assert layer.codebook.shape == (16,)  # 16 codebook entries
        assert layer.boundaries.shape == (15,)  # 15 boundaries for 16 entries

    def test_codebook_boundaries_in_state_dict(self, in_features, out_features):
        """Test that codebook and boundaries are included in state_dict as
        buffers."""
        layer = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

        state_dict = layer.state_dict()

        # Check that codebook and boundaries are in state_dict (as buffers with _ prefix)
        assert '_codebook' in state_dict
        assert '_boundaries' in state_dict

        # Check shapes
        assert state_dict['_codebook'].shape == (16,)
        assert state_dict['_boundaries'].shape == (15,)

    def test_codebook_boundaries_save_load(self, in_features, out_features):
        """Test that codebook and boundaries can be saved and loaded."""
        layer1 = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

        # Save state dict
        state_dict = layer1.state_dict()

        # Create new layer and load state dict
        layer2 = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )
        layer2.load_state_dict(state_dict)

        # Check that codebook and boundaries are the same
        assert torch.allclose(layer1.codebook, layer2.codebook)
        assert torch.allclose(layer1.boundaries, layer2.boundaries)


class TestMergedTurboQuantLinearParameterRegistration:
    """Test MergedTurboQuantLinear codebook and boundaries parameter
    registration."""

    @pytest.fixture
    def in_features(self):
        yield 128

    @pytest.fixture
    def out_features(self):
        yield 256

    def test_merged_codebook_boundaries_as_buffers(self, in_features, out_features):
        """Test that MergedTurboQuantLinear has codebook and boundaries as
        buffers."""
        # Create two TurboQuantLinear layers
        layer1 = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )
        layer2 = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

        # Create merged layer
        merged = MergedTurboQuantLinear(
            layers=[layer1, layer2],
            out_names=['gate', 'up'],
        )

        # Check that codebook and boundaries are accessible via properties
        assert hasattr(merged, 'codebook')
        assert hasattr(merged, 'boundaries')
        assert isinstance(merged.codebook, torch.Tensor)
        assert isinstance(merged.boundaries, torch.Tensor)

        # Check that they are registered as buffers (not parameters)
        assert '_codebook' in dict(merged.named_buffers())
        assert '_boundaries' in dict(merged.named_buffers())
        assert 'codebook' not in dict(merged.named_parameters())
        assert 'boundaries' not in dict(merged.named_parameters())

    def test_merged_codebook_boundaries_in_state_dict(self, in_features, out_features):
        """Test that MergedTurboQuantLinear codebook and boundaries are in
        state_dict as buffers."""
        # Create two TurboQuantLinear layers
        layer1 = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )
        layer2 = TurboQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

        # Create merged layer
        merged = MergedTurboQuantLinear(
            layers=[layer1, layer2],
            out_names=['gate', 'up'],
        )

        state_dict = merged.state_dict()

        # Check that codebook and boundaries are in state_dict (as buffers with _ prefix)
        assert '_codebook' in state_dict
        assert '_boundaries' in state_dict


# =============================================================================
# MergedTurboQuantLinear Correctness Tests
# =============================================================================

class TestMergedTurboQuantLinearCorrectness:
    """Test MergedTurboQuantLinear correctness."""

    @pytest.fixture
    def in_features(self):
        yield 1024

    @pytest.fixture
    def gate_out(self):
        yield 3072

    @pytest.fixture
    def up_out(self):
        yield 3072

    @pytest.fixture
    def batch_seq(self):
        yield (2, 4)

    def test_weight_loader_with_shard_id(self, in_features, gate_out, up_out):
        """Test weight_loader handles shard_id correctly (required for
        stacked_params_mapping)."""
        torch.manual_seed(42)

        layer1 = TurboQuantLinear(
            in_features=in_features,
            out_features=gate_out,
            bias=False,
            colwise=True,
            is_tp=False,
            all_reduce=False,
            bit_width=4,
            group_size=128,
            layer_type='mlp',
        ).cuda()

        layer2 = TurboQuantLinear(
            in_features=in_features,
            out_features=up_out,
            bias=False,
            colwise=True,
            is_tp=False,
            all_reduce=False,
            bit_width=4,
            group_size=128,
            layer_type='mlp',
        ).cuda()

        merged = MergedTurboQuantLinear(
            layers=[layer1, layer2],
            out_names=['gate', 'up'],
            is_tp=False,
            all_reduce=False,
            layer_type='mlp',
        ).cuda()

        w_gate = torch.randn(gate_out, in_features, dtype=torch.bfloat16, device='cuda') * 0.02
        w_up = torch.randn(up_out, in_features, dtype=torch.bfloat16, device='cuda') * 0.02

        merged.weight_loader(merged.weight, w_gate, shard_id=0)
        merged.weight_loader(merged.weight, w_up, shard_id=1)

        W_deq = merged._dequantize_weight().float()

        assert W_deq.shape == (gate_out + up_out, in_features), \
            f'Expected shape ({gate_out + up_out}, {in_features}), got {W_deq.shape}'

        assert not torch.isnan(W_deq).any(), 'Dequantized weight contains NaN'
        assert not torch.isinf(W_deq).any(), 'Dequantized weight contains Inf'

        print(f'  weight_loader with shard_id: OK, shape={W_deq.shape}')

    def test_dequantized_weight_shape(self, in_features, gate_out, up_out):
        """Test dequantized weight has correct shape."""
        torch.manual_seed(42)

        layer1 = TurboQuantLinear(
            in_features=in_features,
            out_features=gate_out,
            bias=False,
        ).cuda()
        layer2 = TurboQuantLinear(
            in_features=in_features,
            out_features=up_out,
            bias=False,
        ).cuda()

        merged = MergedTurboQuantLinear(
            layers=[layer1, layer2],
            out_names=['gate', 'up'],
        ).cuda()

        w_gate = torch.randn(gate_out, in_features, dtype=torch.bfloat16, device='cuda')
        w_up = torch.randn(up_out, in_features, dtype=torch.bfloat16, device='cuda')
        merged.weight_loader(merged.weight, w_gate, shard_id=0)
        merged.weight_loader(merged.weight, w_up, shard_id=1)

        W_deq = merged._dequantize_weight()

        expected_shape = (gate_out + up_out, in_features)
        assert W_deq.shape == expected_shape, \
            f'Expected {expected_shape}, got {W_deq.shape}'

        print(f'  dequantized weight shape: {W_deq.shape}')

    def test_forward_output_valid(self, in_features, gate_out, up_out, batch_seq):
        """Test forward output is valid (no NaN/Inf, correct shape)."""
        torch.manual_seed(42)

        batch, seq = batch_seq

        layer1 = TurboQuantLinear(
            in_features=in_features,
            out_features=gate_out,
            bias=False,
        ).cuda()
        layer2 = TurboQuantLinear(
            in_features=in_features,
            out_features=up_out,
            bias=False,
        ).cuda()

        merged = MergedTurboQuantLinear(
            layers=[layer1, layer2],
            out_names=['gate', 'up'],
        ).cuda()

        w_gate = torch.randn(gate_out, in_features, dtype=torch.bfloat16, device='cuda')
        w_up = torch.randn(up_out, in_features, dtype=torch.bfloat16, device='cuda')
        merged.weight_loader(merged.weight, w_gate, shard_id=0)
        merged.weight_loader(merged.weight, w_up, shard_id=1)

        x = torch.randn(batch, seq, in_features, dtype=torch.bfloat16, device='cuda')

        with torch.no_grad():
            out = merged(x)

        expected_shape = (batch, seq, gate_out + up_out)
        assert out.shape == expected_shape, \
            f'Expected shape {expected_shape}, got {out.shape}'

        assert not torch.isnan(out).any(), 'Forward output contains NaN'
        assert not torch.isinf(out).any(), 'Forward output contains Inf'

        assert not torch.all(out == 0), 'Forward output is all zeros'

        print(f'  forward output: shape={out.shape}, mean={out.float().mean().item():.4f}')

    def test_forward_gate_up_slices_valid(self, in_features, gate_out, up_out, batch_seq):
        """Test forward output gate and up slices are both valid."""
        torch.manual_seed(42)

        batch, seq = batch_seq

        layer1 = TurboQuantLinear(
            in_features=in_features,
            out_features=gate_out,
            bias=False,
        ).cuda()
        layer2 = TurboQuantLinear(
            in_features=in_features,
            out_features=up_out,
            bias=False,
        ).cuda()

        merged = MergedTurboQuantLinear(
            layers=[layer1, layer2],
            out_names=['gate', 'up'],
        ).cuda()

        w_gate = torch.randn(gate_out, in_features, dtype=torch.bfloat16, device='cuda')
        w_up = torch.randn(up_out, in_features, dtype=torch.bfloat16, device='cuda')
        merged.weight_loader(merged.weight, w_gate, shard_id=0)
        merged.weight_loader(merged.weight, w_up, shard_id=1)

        x = torch.randn(batch, seq, in_features, dtype=torch.bfloat16, device='cuda')

        with torch.no_grad():
            out = merged(x)

        gate_out_tensor = out[..., :gate_out]
        up_out_tensor = out[..., gate_out:gate_out + up_out]

        assert not torch.all(gate_out_tensor == 0), 'Gate output is all zeros'
        assert not torch.all(up_out_tensor == 0), 'Up output is all zeros'

        assert gate_out_tensor.abs().max().item() < 500, 'Gate output magnitude too large'
        assert up_out_tensor.abs().max().item() < 500, 'Up output magnitude too large'

        print(f'  gate slice: mean={gate_out_tensor.float().mean().item():.4f}, '
              f'max={gate_out_tensor.abs().max().item():.2f}')
        print(f'  up slice: mean={up_out_tensor.float().mean().item():.4f}, '
              f'max={up_out_tensor.abs().max().item():.2f}')
