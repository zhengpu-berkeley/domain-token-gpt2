"""
Unit tests for SFT dataset formatting.

Tests:
- EOS token is present in formatted examples
- Labels are correctly masked (padding = -100, content = token_ids)
- Mul-token injection in training data

Key lesson captured: When pad_token_id == eos_token_id (common in GPT-2),
using DataCollatorForLanguageModeling will mask ALL occurrences of that token
in labels, including the real EOS we want the model to learn. Use 
default_data_collator and set labels=-100 for padding via attention_mask instead.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEOSTokenHandling:
    """
    Test that EOS tokens are properly handled in dataset formatting.
    
    The critical bug: GPT-2's pad_token_id == eos_token_id == 50256.
    If we use DataCollatorForLanguageModeling, it masks all pad_token_id
    occurrences in labels, which kills EOS learning.
    """

    def test_eos_in_formatted_text(self, sample_gsm8k_examples):
        """EOS token should be appended to formatted examples."""
        # Simulate formatting logic from SFT scripts
        eos_token = "<|endoftext|>"
        
        for example in sample_gsm8k_examples:
            formatted = f"User: {example['question']}\nAssistant: {example['answer']}{eos_token}"
            assert formatted.endswith(eos_token)
            assert eos_token in formatted

    def test_label_masking_preserves_eos(self):
        """Labels should NOT mask the EOS token (only padding)."""
        # Simulate the correct label masking logic
        # input_ids: [token, token, token, EOS, PAD, PAD]
        # attention:  [1,     1,     1,     1,   0,   0]
        # labels:    [token, token, token, EOS, -100, -100]
        
        input_ids = [100, 200, 300, 50256, 50256, 50256]  # EOS=50256, also used as PAD
        attention_mask = [1, 1, 1, 1, 0, 0]  # 4 real tokens, 2 padding
        
        # Correct masking: use attention_mask, not token_id matching
        labels = [
            -100 if mask == 0 else token
            for token, mask in zip(input_ids, attention_mask)
        ]
        
        assert labels == [100, 200, 300, 50256, -100, -100]
        # Key assertion: the FIRST 50256 (real EOS) is preserved in labels
        assert labels[3] == 50256
        # Only the padding 50256s are masked
        assert labels[4] == -100
        assert labels[5] == -100

    def test_wrong_label_masking_masks_eos(self):
        """Demonstrate the bug: token-based masking kills EOS learning."""
        # This is the WRONG approach (what DataCollatorForLanguageModeling does)
        input_ids = [100, 200, 300, 50256, 50256, 50256]
        pad_token_id = 50256
        
        # Wrong: mask by token_id match
        labels_wrong = [
            -100 if token == pad_token_id else token
            for token in input_ids
        ]
        
        # BUG: ALL 50256 tokens are masked, including the real EOS
        assert labels_wrong == [100, 200, 300, -100, -100, -100]
        # The model never learns to predict EOS!
        assert labels_wrong[3] == -100  # This is the bug


class TestMulTokenInjection:
    """Test that mul-tokens are properly injected into training data."""

    def test_injection_in_answer(self, sample_mul_expressions):
        """Mul-tokens should be injected into answer text."""
        from tokenizer.inject_mul import create_injector
        
        injector = create_injector("weak")
        
        for input_expr, expected in sample_mul_expressions:
            result = injector.inject(input_expr)
            assert result == expected, f"Expected '{expected}' for '{input_expr}', got '{result}'"

    def test_injection_preserves_context(self):
        """Injection should preserve surrounding text."""
        from tokenizer.inject_mul import create_injector
        
        injector = create_injector("weak")
        
        text = "The answer is 6*9 which equals 54."
        result = injector.inject(text)
        
        assert "<MUL_6_9_54>" in result
        assert "The answer is" in result
        assert "which equals 54." in result

    def test_injection_in_gsm8k_format(self, sample_gsm8k_examples):
        """Test injection in full GSM8K-style answers."""
        from tokenizer.inject_mul import create_injector
        
        injector = create_injector("weak")
        
        # Create a sample with multiplication
        text = "Janet sells 9 * 2 = 18 eggs. #### 18"
        result = injector.inject(text)
        
        assert "<MUL_2_9_18>" in result  # Canonicalized (2 < 9)
        assert "#### 18" in result


class TestDatasetFormatting:
    """Test complete dataset formatting pipeline."""

    def test_tulu_message_format(self):
        """Test Tulu-style message formatting."""
        # Simulate Tulu format conversion
        messages = [
            {"role": "user", "content": "What is 7 times 8?"},
            {"role": "assistant", "content": "7 times 8 equals 56."},
        ]
        
        eos_token = "<|endoftext|>"
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"Assistant: {msg['content']}")
        
        text = "\n".join(parts) + eos_token
        
        assert "User: What is 7 times 8?" in text
        assert "Assistant: 7 times 8 equals 56." in text
        assert text.endswith(eos_token)

    def test_tinygsm_format(self, sample_tinygsm_examples):
        """Test TinyGSM-style formatting."""
        eos_token = "<|endoftext|>"
        
        for example in sample_tinygsm_examples:
            text = f"User: {example['question']}\nAssistant: {example['solution']}{eos_token}"
            
            assert text.startswith("User:")
            assert "Assistant:" in text
            assert "####" in text
            assert text.endswith(eos_token)

    def test_tokenization_roundtrip(self):
        """Test that formatted text tokenizes and decodes correctly."""
        from tokenizer.gpt2_tiktoken import create_tokenizer
        
        tokenizer = create_tokenizer("mul_tokens")
        
        text = "User: What is 6*9?\nAssistant: <MUL_6_9_54>. The answer is 54.\n#### 54"
        
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text
        
        # Verify mul-token is a single token
        mul_count = tokenizer.count_mul_tokens(tokens)
        assert mul_count == 1


class TestDataCollatorChoice:
    """Document why we use default_data_collator instead of DataCollatorForLanguageModeling."""

    def test_default_collator_preserves_labels(self):
        """default_data_collator should not modify labels."""
        from transformers import default_data_collator
        import torch
        
        # Pre-computed labels with -100 for padding
        batch = [
            {
                "input_ids": torch.tensor([100, 200, 50256, 50256]),
                "attention_mask": torch.tensor([1, 1, 1, 0]),
                "labels": torch.tensor([100, 200, 50256, -100]),  # EOS preserved!
            }
        ]
        
        collated = default_data_collator(batch)
        
        # Labels should be unchanged
        assert collated["labels"][0, 2].item() == 50256  # EOS preserved
        assert collated["labels"][0, 3].item() == -100  # Padding masked

    def test_lm_collator_masks_eos_bug(self):
        """DataCollatorForLanguageModeling masks EOS when pad==eos (the bug)."""
        from transformers import DataCollatorForLanguageModeling, GPT2TokenizerFast
        import torch
        
        # This test documents the bug behavior
        # We don't actually run it because it requires a full tokenizer setup,
        # but the logic is: DCFLM replaces all pad_token_id with -100 in labels
        
        # Pseudo-code of the bug:
        # labels[labels == tokenizer.pad_token_id] = -100
        # When pad_token_id == eos_token_id, this masks the real EOS too!
        
        # The fix: use default_data_collator and pre-compute labels ourselves
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

