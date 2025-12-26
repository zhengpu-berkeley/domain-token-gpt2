"""
Unit tests for the tokenizer module.

Tests:
- MulFactTokens: generation, canonicalization, ID assignment
- GPT2TokenizerWithMulFacts: encode/decode, roundtrip, single-token behavior
"""

import pytest

from tokenizer.mul_facts import (
    MulFactTokens,
    MulToken,
    canonicalize,
    get_mul_token,
    get_mul_token_range,
)
from tokenizer.gpt2_tiktoken import (
    GPT2TokenizerWithMulFacts,
    create_tokenizer,
)


class TestMulFactTokens:
    """Tests for MulFactTokens class."""
    
    def test_token_count(self):
        """Should generate 45 canonicalized tokens for 1-9 range."""
        mul_tokens = MulFactTokens()
        # 1x1, 1x2, ..., 1x9 = 9
        # 2x2, 2x3, ..., 2x9 = 8
        # ...
        # 9x9 = 1
        # Total = 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 45
        assert mul_tokens.num_tokens == 45
    
    def test_vocab_size(self):
        """Vocab size should be base + num_mul_tokens."""
        mul_tokens = MulFactTokens(base_vocab_size=50304)
        assert mul_tokens.vocab_size == 50304 + 45
    
    def test_token_id_range(self):
        """Token IDs should be contiguous starting from base_vocab_size."""
        mul_tokens = MulFactTokens(base_vocab_size=50304)
        first, last = mul_tokens.token_id_range
        assert first == 50304
        assert last == 50304 + 44  # 45 tokens, 0-indexed
    
    def test_token_string_format(self):
        """Token strings should follow <MUL_a_b_c> format."""
        mul_tokens = MulFactTokens()
        for token in mul_tokens.all_tokens:
            assert token.token_str.startswith("<MUL_")
            assert token.token_str.endswith(">")
            # Parse and verify
            parts = token.token_str[5:-1].split("_")
            assert len(parts) == 3
            a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
            assert a == token.a
            assert b == token.b
            assert c == token.c
            assert a * b == c
    
    def test_canonicalization(self):
        """Commutative pairs should map to same token."""
        mul_tokens = MulFactTokens()
        
        # 6*9 and 9*6 should give same token
        token_6_9 = mul_tokens.get_token_for_fact(6, 9)
        token_9_6 = mul_tokens.get_token_for_fact(9, 6)
        assert token_6_9 == token_9_6
        assert token_6_9.a == 6
        assert token_6_9.b == 9
        assert token_6_9.c == 54
    
    def test_canonicalize_function(self):
        """canonicalize should order a <= b."""
        assert canonicalize(6, 9) == (6, 9)
        assert canonicalize(9, 6) == (6, 9)
        assert canonicalize(3, 3) == (3, 3)
        assert canonicalize(1, 9) == (1, 9)
        assert canonicalize(9, 1) == (1, 9)
    
    def test_out_of_range(self):
        """Out-of-range operands should return None."""
        mul_tokens = MulFactTokens()
        
        assert mul_tokens.get_token_for_fact(0, 5) is None
        assert mul_tokens.get_token_for_fact(10, 5) is None
        assert mul_tokens.get_token_for_fact(5, 10) is None
        assert mul_tokens.get_token_for_fact(-1, 3) is None
    
    def test_id_lookup(self):
        """Should be able to look up tokens by ID."""
        mul_tokens = MulFactTokens()
        
        for token in mul_tokens.all_tokens:
            token_id = mul_tokens.get_token_id(token.token_str)
            assert token_id is not None
            
            retrieved = mul_tokens.get_token_by_id(token_id)
            assert retrieved == token
    
    def test_is_mul_token_id(self):
        """Should correctly identify mul token IDs."""
        mul_tokens = MulFactTokens(base_vocab_size=50304)
        
        # Valid mul token IDs
        assert mul_tokens.is_mul_token_id(50304)
        assert mul_tokens.is_mul_token_id(50348)
        assert mul_tokens.is_mul_token_id(50320)
        
        # Invalid (base vocab or beyond)
        assert not mul_tokens.is_mul_token_id(0)
        assert not mul_tokens.is_mul_token_id(50303)
        assert not mul_tokens.is_mul_token_id(50349)
        assert not mul_tokens.is_mul_token_id(100000)


class TestGPT2TokenizerWithMulFacts:
    """Tests for GPT2TokenizerWithMulFacts class."""
    
    def test_baseline_vocab_size(self):
        """Baseline tokenizer should have same vocab_size as mul-token version."""
        baseline = create_tokenizer("baseline")
        mul_tok = create_tokenizer("mul_tokens")
        assert baseline.vocab_size == mul_tok.vocab_size
    
    def test_baseline_no_mul_tokens(self):
        """Baseline should never produce mul token IDs."""
        tokenizer = create_tokenizer("baseline")
        
        test_texts = [
            "Hello world",
            "<MUL_6_9_54>",
            "Calculate <MUL_3_4_12> please.",
        ]
        
        for text in test_texts:
            tokens = tokenizer.encode(text)
            mul_count = tokenizer.count_mul_tokens(tokens)
            assert mul_count == 0, f"Baseline produced mul tokens for: {text}"
    
    def test_mul_token_single_id(self):
        """Mul-fact token should encode to exactly one token ID."""
        tokenizer = create_tokenizer("mul_tokens")
        
        # Test several mul tokens
        test_cases = [
            ("<MUL_1_1_1>", 50304),
            ("<MUL_6_9_54>", 50342),
            ("<MUL_9_9_81>", 50348),
        ]
        
        for token_str, expected_id in test_cases:
            tokens = tokenizer.encode(token_str)
            assert len(tokens) == 1, f"{token_str} should be 1 token, got {len(tokens)}"
            assert tokens[0] == expected_id, f"{token_str} should be ID {expected_id}, got {tokens[0]}"
    
    def test_roundtrip_regular_text(self):
        """Regular text should roundtrip correctly."""
        for condition in ["baseline", "mul_tokens"]:
            tokenizer = create_tokenizer(condition)
            
            test_texts = [
                "Hello, world!",
                "The quick brown fox jumps over the lazy dog.",
                "1 + 2 = 3",
                "αβγδ",  # Unicode
            ]
            
            for text in test_texts:
                tokens = tokenizer.encode(text)
                decoded = tokenizer.decode(tokens)
                assert decoded == text, f"Roundtrip failed for {condition}: '{text}' -> '{decoded}'"
    
    def test_roundtrip_mul_tokens(self):
        """Mul-fact tokens should roundtrip correctly in mul_tokens mode."""
        tokenizer = create_tokenizer("mul_tokens")
        
        test_texts = [
            "<MUL_6_9_54>",
            "The answer is <MUL_3_4_12>.",
            "Calculate <MUL_1_1_1> and <MUL_9_9_81>.",
        ]
        
        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Roundtrip failed: '{text}' -> '{decoded}'"
    
    def test_roundtrip_mul_tokens_baseline(self):
        """Mul-fact tokens should roundtrip as regular text in baseline mode."""
        tokenizer = create_tokenizer("baseline")
        
        text = "<MUL_6_9_54>"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        # Should still decode correctly, just via regular BPE
        assert decoded == text
    
    def test_metadata(self):
        """Metadata should contain expected fields."""
        tokenizer = create_tokenizer("mul_tokens")
        metadata = tokenizer.get_metadata()
        
        assert "vocab_size" in metadata
        assert "base_vocab_size" in metadata
        assert "enable_mul_tokens" in metadata
        assert "mul_token_id_start" in metadata
        assert "mul_token_id_end" in metadata
        assert "num_mul_tokens" in metadata
        
        assert metadata["enable_mul_tokens"] is True
        assert metadata["num_mul_tokens"] == 45
    
    def test_token_stats(self):
        """get_mul_token_stats should return correct counts."""
        tokenizer = create_tokenizer("mul_tokens")
        
        text = "Result: <MUL_6_9_54> and <MUL_3_3_9>."
        tokens = tokenizer.encode(text)
        stats = tokenizer.get_mul_token_stats(tokens)
        
        assert stats["mul_tokens"] == 2
        assert stats["total_tokens"] > 2
        assert stats["regular_tokens"] == stats["total_tokens"] - 2
        assert 0 < stats["mul_token_ratio"] < 1
    
    def test_batch_encode_decode(self):
        """Batch operations should work correctly."""
        tokenizer = create_tokenizer("mul_tokens")
        
        texts = [
            "Hello",
            "<MUL_2_3_6>",
            "Test <MUL_5_5_25> here",
        ]
        
        token_lists = tokenizer.encode_batch(texts)
        decoded = tokenizer.decode_batch(token_lists)
        
        assert len(token_lists) == 3
        assert len(decoded) == 3
        for orig, dec in zip(texts, decoded):
            assert orig == dec


class TestCreateTokenizer:
    """Tests for create_tokenizer factory function."""
    
    def test_baseline_condition(self):
        """'baseline' should create tokenizer with mul_tokens disabled."""
        tokenizer = create_tokenizer("baseline")
        assert tokenizer.enable_mul_tokens is False
    
    def test_mul_tokens_condition(self):
        """'mul_tokens' should create tokenizer with mul_tokens enabled."""
        tokenizer = create_tokenizer("mul_tokens")
        assert tokenizer.enable_mul_tokens is True
    
    def test_invalid_condition(self):
        """Invalid condition should raise ValueError."""
        with pytest.raises(ValueError):
            create_tokenizer("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

