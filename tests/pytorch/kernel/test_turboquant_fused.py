"""Tests for TurboQuant fused kernels (turbo_quant_fused.py).

This module tests the Triton fused dequant+matmul kernels that avoid
materializing the full dequantized weight matrix by fusing:
  4-bit unpack → codebook lookup → matmul → norm rescale

in a single kernel launch.
"""

import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.turbo_quant_fused import (
    triton_fused_dequant_matmul,
    turboquant_fused_forward,
)

from .turboquant_utils import (
    make_weight_packed,
    reference_fused_matmul,
    reference_turboquant_forward,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def codebook():
    """4-bit Lloyd-Max codebook (16 entries)."""
    return torch.tensor([
        -2.7943, -2.1384, -1.6884, -1.3222,
        -0.9991, -0.7004, -0.4155, -0.1378,
         0.1378,  0.4155,  0.7004,  0.9991,
         1.3222,  1.6884,  2.1384,  2.7943
    ], dtype=torch.float32, device='cuda')


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestTritonFusedDequantMatmul:
    """Test triton_fused_dequant_matmul kernel."""

    @pytest.mark.parametrize('B,N,K', [
        # decode-like scenarios
        (1, 256, 512),
        (4, 256, 512),
        # small batch prefill
        (16, 128, 512),
        # large batch prefill
        (32, 64, 512),
    ])
    def test_correctness_vs_reference(self, B, N, K, codebook):
        """Verify fused kernel matches reference implementation."""
        torch.manual_seed(42)
        group_size = 128

        # Create inputs
        x_rot = torch.randn(B, K, dtype=torch.float32, device='cuda')
        weight_packed, weight_norms = make_weight_packed(N, K)

        # Compute via fused kernel
        output_fused = triton_fused_dequant_matmul(
            x_rot=x_rot,
            indices_packed=weight_packed,
            codebook=codebook,
            norms=weight_norms,
            in_features=K,
            group_size=group_size,
        )

        # Compute via reference
        output_ref = reference_fused_matmul(
            x_rot=x_rot,
            indices_packed=weight_packed,
            codebook=codebook,
            norms=weight_norms,
            group_size=group_size,
        )

        # Verify numerical correctness (relaxed tolerance for floating point)
        max_diff = (output_fused - output_ref).abs().max().item()
        torch.testing.assert_close(output_fused, output_ref, atol=0.5, rtol=1e-2)
        print(f'  B={B}, N={N}, K={K}: max_diff={max_diff:.6f}')

    @pytest.mark.parametrize('B,N,K', [
        (1, 256, 512),
        (16, 128, 512),
    ])
    def test_with_sigma_scale(self, B, N, K, codebook):
        """Test with explicit sigma_scale parameter."""
        torch.manual_seed(42)
        group_size = 128

        x_rot = torch.randn(B, K, dtype=torch.float32, device='cuda')
        weight_packed, weight_norms = make_weight_packed(N, K)

        sigma_scale = 0.1

        output = triton_fused_dequant_matmul(
            x_rot=x_rot,
            indices_packed=weight_packed,
            codebook=codebook,
            norms=weight_norms,
            in_features=K,
            group_size=group_size,
            sigma_scale=sigma_scale,
        )

        assert output.shape == (B, N)
        assert output.dtype == torch.float32
        print(f'  sigma_scale={sigma_scale}: output range=[{output.min():.3f}, {output.max():.3f}]')

    def test_output_shape(self, codebook):
        """Verify output tensor has correct shape."""
        B, N, K = 8, 256, 512
        group_size = 128

        x_rot = torch.randn(B, K, dtype=torch.float32, device='cuda')
        weight_packed, weight_norms = make_weight_packed(N, K)

        output = triton_fused_dequant_matmul(
            x_rot=x_rot,
            indices_packed=weight_packed,
            codebook=codebook,
            norms=weight_norms,
            in_features=K,
            group_size=group_size,
        )

        assert output.shape == (B, N)
        print(f'  output shape: {output.shape}')

    def test_determinism(self, codebook):
        """Verify same inputs produce same outputs."""
        B, N, K = 4, 128, 512
        group_size = 128

        x_rot = torch.randn(B, K, dtype=torch.float32, device='cuda')
        weight_packed, weight_norms = make_weight_packed(N, K)

        out1 = triton_fused_dequant_matmul(
            x_rot=x_rot,
            indices_packed=weight_packed,
            codebook=codebook,
            norms=weight_norms,
            in_features=K,
            group_size=group_size,
        )

        out2 = triton_fused_dequant_matmul(
            x_rot=x_rot,
            indices_packed=weight_packed,
            codebook=codebook,
            norms=weight_norms,
            in_features=K,
            group_size=group_size,
        )

        torch.testing.assert_close(out1, out2)
        print('  determinism: OK')


class TestTurboquantFusedForward:
    """Test turboquant_fused_forward high-level function."""

    @pytest.mark.parametrize('shape', [
        # 2D: (B, K)
        (1, 512),
        (4, 512),
        (16, 512),
        # 3D: (B, seq, K)
        (1, 1, 512),
        (2, 8, 512),
        (4, 4, 512),
    ])
    def test_correctness_vs_reference(self, shape, codebook):
        """Verify full forward matches reference implementation."""
        torch.manual_seed(42)
        in_features = 512
        out_features = 256
        group_size = 128

        # Create inputs
        x = torch.randn(*shape, dtype=torch.float32, device='cuda')
        weight_packed, weight_norms = make_weight_packed(out_features, in_features)

        # Compute via fused forward
        output_fused = turboquant_fused_forward(
            x=x,
            weight_packed=weight_packed,
            codebook=codebook,
            weight_norms=weight_norms,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
        )

        # Compute via reference
        output_ref = reference_turboquant_forward(
            x=x,
            weight_packed=weight_packed,
            codebook=codebook,
            weight_norms=weight_norms,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
        )

        # Verify numerical correctness (relaxed tolerance for floating point)
        max_diff = (output_fused - output_ref).abs().max().item()
        torch.testing.assert_close(output_fused, output_ref, atol=0.5, rtol=1e-2)
        print(f'  shape={shape}: max_diff={max_diff:.6f}')

    @pytest.mark.parametrize('shape', [
        (1, 512),
        (4, 512),
        (2, 8, 512),
    ])
    def test_with_bias(self, shape, codebook):
        """Test with optional bias."""
        torch.manual_seed(42)
        in_features = 512
        out_features = 256
        group_size = 128

        x = torch.randn(*shape, dtype=torch.float32, device='cuda')
        weight_packed, weight_norms = make_weight_packed(out_features, in_features)
        bias = torch.randn(out_features, dtype=torch.float32, device='cuda')

        output = turboquant_fused_forward(
            x=x,
            weight_packed=weight_packed,
            codebook=codebook,
            weight_norms=weight_norms,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            bias=bias,
        )

        # Verify bias is applied
        output_no_bias = turboquant_fused_forward(
            x=x,
            weight_packed=weight_packed,
            codebook=codebook,
            weight_norms=weight_norms,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            bias=None,
        )

        diff = (output - output_no_bias).abs()
        # Bias should be approximately preserved (accounting for dtype conversion)
        assert diff.mean().item() > 0.0, 'bias should affect output'
        print(f'  shape={shape}: bias effect mean={diff.mean().item():.6f}')

    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
    def test_dtype_handling(self, dtype, codebook):
        """Test correct dtype handling for input and output."""
        torch.manual_seed(42)
        in_features = 512
        out_features = 256
        group_size = 128

        x = torch.randn(2, 8, in_features, dtype=dtype, device='cuda')
        weight_packed, weight_norms = make_weight_packed(out_features, in_features)

        output = turboquant_fused_forward(
            x=x,
            weight_packed=weight_packed,
            codebook=codebook,
            weight_norms=weight_norms,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
        )

        # Output dtype should match input (or bfloat16 for float32 input)
        expected_dtype = dtype if dtype != torch.float32 else torch.bfloat16
        assert output.dtype == expected_dtype, f'Expected {expected_dtype}, got {output.dtype}'
        print(f'  input dtype={dtype}, output dtype={output.dtype}')

    def test_output_shape_2d(self, codebook):
        """Verify correct output shape for 2D input."""
        torch.manual_seed(42)
        in_features = 512
        out_features = 256
        group_size = 128

        B = 4
        x = torch.randn(B, in_features, dtype=torch.float32, device='cuda')
        weight_packed, weight_norms = make_weight_packed(out_features, in_features)

        output = turboquant_fused_forward(
            x=x,
            weight_packed=weight_packed,
            codebook=codebook,
            weight_norms=weight_norms,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
        )

        assert output.shape == (B, out_features)
        print(f'  2D output shape: {output.shape}')

    def test_output_shape_3d(self, codebook):
        """Verify correct output shape for 3D input."""
        torch.manual_seed(42)
        in_features = 512
        out_features = 256
        group_size = 128

        B, seq = 2, 8
        x = torch.randn(B, seq, in_features, dtype=torch.float32, device='cuda')
        weight_packed, weight_norms = make_weight_packed(out_features, in_features)

        output = turboquant_fused_forward(
            x=x,
            weight_packed=weight_packed,
            codebook=codebook,
            weight_norms=weight_norms,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
        )

        assert output.shape == (B, seq, out_features)
        print(f'  3D output shape: {output.shape}')

    def test_invalid_input_dim(self, codebook):
        """Test that invalid input dimensions raise ValueError."""
        torch.manual_seed(42)
        in_features = 512
        out_features = 256
        group_size = 128

        # 1D input should raise
        x = torch.randn(in_features, dtype=torch.float32, device='cuda')
        weight_packed, weight_norms = make_weight_packed(out_features, in_features)

        with pytest.raises(ValueError, match='Expected 2-D or 3-D input'):
            turboquant_fused_forward(
                x=x,
                weight_packed=weight_packed,
                codebook=codebook,
                weight_norms=weight_norms,
                in_features=in_features,
                out_features=out_features,
                group_size=group_size,
            )
        print('  invalid input dim: correctly raised ValueError')

    def test_determinism(self, codebook):
        """Verify same inputs produce same outputs."""
        torch.manual_seed(42)
        in_features = 512
        out_features = 256
        group_size = 128

        x = torch.randn(2, 8, in_features, dtype=torch.float32, device='cuda')
        weight_packed, weight_norms = make_weight_packed(out_features, in_features)

        out1 = turboquant_fused_forward(
            x=x,
            weight_packed=weight_packed,
            codebook=codebook,
            weight_norms=weight_norms,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
        )

        out2 = turboquant_fused_forward(
            x=x,
            weight_packed=weight_packed,
            codebook=codebook,
            weight_norms=weight_norms,
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
        )

        torch.testing.assert_close(out1, out2)
        print('  determinism: OK')
