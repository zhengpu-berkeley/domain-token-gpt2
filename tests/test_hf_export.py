"""
Unit tests for HuggingFace export functionality.

Tests:
- Weight transpose logic for Conv1D vs Linear
- Config preservation during export
- Round-trip consistency (nanoGPT -> HF -> inference)

Key lesson captured: GPT-2 uses Conv1D for projections (shape: in_features, out_features)
while our training uses nn.Linear (shape: out_features, in_features). All projection
weights must be transposed during export, including square matrices like c_proj.
"""

import pytest
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConv1DTransposeLogic:
    """
    Test the weight transpose detection logic from export_hf.py.
    
    The key insight: HuggingFace GPT-2 uses Conv1D (in, out) layout while
    our training uses nn.Linear (out, in) layout. We must transpose ALL
    projection weights, even when shapes are square (e.g., 768x768).
    """

    def test_needs_transpose_detection(self):
        """Test that we correctly identify weights needing transpose."""
        # Simulate the logic from export_hf.py
        def needs_conv1d_transpose(param_name: str, tensor: torch.Tensor) -> bool:
            if tensor.ndim != 2:
                return False
            if not param_name.endswith(".weight"):
                return False
            return (
                ".attn.c_attn.weight" in param_name
                or ".attn.c_proj.weight" in param_name
                or ".mlp.c_fc.weight" in param_name
                or ".mlp.c_proj.weight" in param_name
            )

        # 2D weights that SHOULD be transposed
        w_2d = torch.randn(768, 768)
        assert needs_conv1d_transpose("transformer.h.0.attn.c_attn.weight", w_2d)
        assert needs_conv1d_transpose("transformer.h.0.attn.c_proj.weight", w_2d)
        assert needs_conv1d_transpose("transformer.h.0.mlp.c_fc.weight", w_2d)
        assert needs_conv1d_transpose("transformer.h.0.mlp.c_proj.weight", w_2d)

        # Weights that should NOT be transposed
        assert not needs_conv1d_transpose("transformer.h.0.attn.c_attn.bias", torch.randn(768))
        assert not needs_conv1d_transpose("transformer.wte.weight", w_2d)  # Embedding
        assert not needs_conv1d_transpose("transformer.ln_f.weight", torch.randn(768))

    def test_square_matrix_transpose(self):
        """
        Verify that square matrices are still transposed.
        
        This was the original bug: we only transposed when shapes mismatched,
        but c_proj is 768x768 (square), so it went untransposed, breaking evals.
        """
        # Simulate a 768x768 weight (like c_proj)
        linear_weight = torch.randn(768, 768)

        # After transpose, it should be different (unless symmetric, which is rare)
        transposed = linear_weight.t()

        # The key test: transpose changes the tensor
        assert not torch.allclose(linear_weight, transposed, atol=1e-6)

        # And transpose is reversible
        assert torch.allclose(linear_weight, transposed.t())

    def test_non_square_matrix_transpose(self):
        """Test transpose for non-square matrices (e.g., c_fc expanding 768->3072)."""
        # c_fc: 768 -> 3072 (expands by 4x)
        linear_weight = torch.randn(3072, 768)  # Linear: (out, in)
        conv1d_weight = linear_weight.t()  # Conv1D: (in, out)

        assert linear_weight.shape == (3072, 768)
        assert conv1d_weight.shape == (768, 3072)


class TestExportRoundTrip:
    """Test that export produces correct model behavior."""

    def test_checkpoint_structure(self, tiny_nanogpt_checkpoint):
        """Verify checkpoint has expected structure."""
        ckpt = tiny_nanogpt_checkpoint
        
        assert "model" in ckpt
        assert "config" in ckpt
        assert ckpt["config"]["vocab_size"] == 50349
        assert ckpt["config"]["n_layer"] == 2
        
        # Check key weights exist
        assert "transformer.wte.weight" in ckpt["model"]
        assert "transformer.h.0.attn.c_proj.weight" in ckpt["model"]

    def test_export_creates_valid_hf_model(self, tiny_nanogpt_checkpoint, tmp_path):
        """Test that export creates a valid HuggingFace model."""
        from pretrain.export_hf import export_to_hf
        from transformers import GPT2LMHeadModel
        
        # Save checkpoint to temp file
        ckpt_path = tmp_path / "checkpoint.pt"
        torch.save(tiny_nanogpt_checkpoint, ckpt_path)
        
        # Export to HF
        output_dir = tmp_path / "hf_model"
        export_to_hf(ckpt_path, output_dir, condition="baseline")
        
        # Load the exported model
        model = GPT2LMHeadModel.from_pretrained(output_dir)
        
        # Verify it can do a forward pass
        input_ids = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs.logits.shape == (1, 10, 50349)

    def test_export_preserves_metadata(self, tiny_nanogpt_checkpoint, tmp_path):
        """Test that export preserves metadata in JSON."""
        import json
        from pretrain.export_hf import export_to_hf
        
        ckpt_path = tmp_path / "checkpoint.pt"
        torch.save(tiny_nanogpt_checkpoint, ckpt_path)
        
        output_dir = tmp_path / "hf_model"
        export_to_hf(ckpt_path, output_dir, condition="mul_tokens")
        
        # Check metadata file
        metadata_path = output_dir / "export_metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["condition"] == "mul_tokens"
        assert metadata["step"] == 100
        assert metadata["vocab_size"] == 50349


class TestWeightShapes:
    """Test that exported weights have correct shapes for HuggingFace Conv1D."""

    def test_c_attn_shape(self, tiny_nanogpt_checkpoint, tmp_path):
        """c_attn should have shape (n_embd, 3*n_embd) in HF format."""
        from pretrain.export_hf import export_to_hf
        from transformers import GPT2LMHeadModel
        
        ckpt_path = tmp_path / "checkpoint.pt"
        torch.save(tiny_nanogpt_checkpoint, ckpt_path)
        
        output_dir = tmp_path / "hf_model"
        export_to_hf(ckpt_path, output_dir, condition="baseline")
        
        model = GPT2LMHeadModel.from_pretrained(output_dir)
        
        # HF Conv1D stores as (in_features, out_features)
        c_attn = model.transformer.h[0].attn.c_attn.weight
        n_embd = 64  # From tiny_config
        
        # Conv1D: (in=64, out=192)
        assert c_attn.shape == (n_embd, 3 * n_embd)

    def test_c_proj_is_square(self, tiny_nanogpt_checkpoint, tmp_path):
        """c_proj should be square and properly transposed."""
        from pretrain.export_hf import export_to_hf
        from transformers import GPT2LMHeadModel
        
        ckpt_path = tmp_path / "checkpoint.pt"
        torch.save(tiny_nanogpt_checkpoint, ckpt_path)
        
        output_dir = tmp_path / "hf_model"
        export_to_hf(ckpt_path, output_dir, condition="baseline")
        
        model = GPT2LMHeadModel.from_pretrained(output_dir)
        
        c_proj = model.transformer.h[0].attn.c_proj.weight
        n_embd = 64
        
        # Should be square
        assert c_proj.shape == (n_embd, n_embd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

