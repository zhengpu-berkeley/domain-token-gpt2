"""
Unit tests for the multiplication expression injector.

Tests:
- Pattern matching for various multiplication formats
- Canonicalization of commutative expressions
- Weak vs strict mode behavior
- Edge cases and out-of-range values
"""

import pytest

from tokenizer.inject_mul import (
    MulExpressionInjector,
    create_injector,
)
from tokenizer.gpt2_tiktoken import create_tokenizer


class TestMulExpressionInjector:
    """Tests for MulExpressionInjector class."""
    
    def test_asterisk_multiplication(self):
        """Should detect a*b patterns."""
        injector = create_injector("weak")
        
        assert injector.inject("6*9") == "<MUL_6_9_54>"
        assert injector.inject("3*4") == "<MUL_3_4_12>"
        assert injector.inject("1*1") == "<MUL_1_1_1>"
    
    def test_asterisk_with_spaces(self):
        """Should detect a * b patterns with spaces."""
        injector = create_injector("weak")
        
        assert injector.inject("6 * 9") == "<MUL_6_9_54>"
        assert injector.inject("6  *  9") == "<MUL_6_9_54>"
        assert injector.inject("3 *4") == "<MUL_3_4_12>"
        assert injector.inject("3* 4") == "<MUL_3_4_12>"
    
    def test_unicode_multiplication(self):
        """Should detect a×b patterns (unicode ×)."""
        injector = create_injector("weak")
        
        assert injector.inject("6×9") == "<MUL_6_9_54>"
        assert injector.inject("6 × 9") == "<MUL_6_9_54>"
    
    def test_lowercase_x_multiplication(self):
        """Should detect axb patterns (lowercase x)."""
        injector = create_injector("weak")
        
        assert injector.inject("6x9") == "<MUL_6_9_54>"
        assert injector.inject("3 x 4") == "<MUL_3_4_12>"
    
    def test_lowercase_x_not_variable(self):
        """Should not match variable-like patterns like '2x'."""
        injector = create_injector("weak")
        
        # "2x" followed by letter should not match
        assert injector.inject("2xy") == "2xy"
        assert injector.inject("solve for 2x") == "solve for 2x"
    
    def test_canonicalization(self):
        """Commutative expressions should produce same token."""
        injector = create_injector("weak")
        
        # 6*9 and 9*6 should both become <MUL_6_9_54>
        assert injector.inject("6*9") == "<MUL_6_9_54>"
        assert injector.inject("9*6") == "<MUL_6_9_54>"
        
        # 3*7 and 7*3
        assert injector.inject("3*7") == "<MUL_3_7_21>"
        assert injector.inject("7*3") == "<MUL_3_7_21>"
    
    def test_with_result(self):
        """Should handle a*b=c patterns."""
        injector = create_injector("weak")
        
        assert injector.inject("6*9=54") == "<MUL_6_9_54>"
        assert injector.inject("6 * 9 = 54") == "<MUL_6_9_54>"
        assert injector.inject("3×4=12") == "<MUL_3_4_12>"
    
    def test_wrong_result_not_injected(self):
        """Should not inject when result is wrong (verify_result=True)."""
        injector = create_injector("weak")
        
        # 6*9=55 is wrong, should not be changed
        assert injector.inject("6*9=55") == "6*9=55"
        assert injector.inject("3*4=13") == "3*4=13"
    
    def test_out_of_range_not_injected(self):
        """Should not inject for out-of-range operands."""
        injector = create_injector("weak")
        
        # 10 and above are out of range
        assert injector.inject("10*5") == "10*5"
        assert injector.inject("5*10") == "5*10"
        assert injector.inject("10*10") == "10*10"
        assert injector.inject("0*5") == "0*5"
    
    def test_strict_mode_requires_result(self):
        """Strict mode should only inject when =c is present."""
        injector = create_injector("strict")
        
        # Without result: not injected
        assert injector.inject("6*9") == "6*9"
        assert injector.inject("3 × 4") == "3 × 4"
        
        # With result: injected
        assert injector.inject("6*9=54") == "<MUL_6_9_54>"
        assert injector.inject("3 × 4 = 12") == "<MUL_3_4_12>"
    
    def test_multiple_expressions(self):
        """Should handle multiple expressions in one text."""
        injector = create_injector("weak")
        
        text = "Calculate 6*9 and 3*4."
        expected = "Calculate <MUL_6_9_54> and <MUL_3_4_12>."
        assert injector.inject(text) == expected
    
    def test_mixed_inject_and_keep(self):
        """Should inject valid and keep invalid expressions."""
        injector = create_injector("weak")
        
        text = "Valid: 6*9, Invalid: 10*5, Wrong: 3*4=15"
        result = injector.inject(text)
        
        assert "<MUL_6_9_54>" in result
        assert "10*5" in result
        assert "3*4=15" in result
    
    def test_in_sentence_context(self):
        """Should work correctly within sentence context."""
        injector = create_injector("weak")
        
        text = "The product of 6*9 is 54."
        expected = "The product of <MUL_6_9_54> is 54."
        assert injector.inject(text) == expected
    
    def test_batch_inject(self):
        """inject_batch should process multiple texts."""
        injector = create_injector("weak")
        
        texts = ["6*9", "3*4", "10*5"]
        results = injector.inject_batch(texts)
        
        assert results == ["<MUL_6_9_54>", "<MUL_3_4_12>", "10*5"]
    
    def test_find_expressions(self):
        """find_expressions should return detailed info."""
        injector = create_injector("weak")
        
        text = "Calculate 6*9 and 10*5."
        exprs = injector.find_expressions(text)
        
        assert len(exprs) == 2
        
        # First expression (6*9)
        assert exprs[0]["a"] == 6
        assert exprs[0]["b"] == 9
        assert exprs[0]["would_inject"] is True
        assert exprs[0]["token"] == "<MUL_6_9_54>"
        
        # Second expression (10*5) - out of range
        assert exprs[1]["a"] == 10
        assert exprs[1]["b"] == 5
        assert exprs[1]["would_inject"] is False
        assert exprs[1]["token"] is None


class TestInjectorWithTokenizer:
    """Integration tests: injector + tokenizer working together."""
    
    def test_inject_then_encode_single_token(self):
        """Injected text should encode mul-fact as single token."""
        injector = create_injector("weak")
        tokenizer = create_tokenizer("mul_tokens")
        
        raw_text = "What is 6*9?"
        injected = injector.inject(raw_text)
        tokens = tokenizer.encode(injected)
        
        # Count mul tokens
        mul_count = tokenizer.count_mul_tokens(tokens)
        assert mul_count == 1
    
    def test_inject_encode_decode_roundtrip(self):
        """Full pipeline: inject -> encode -> decode."""
        injector = create_injector("weak")
        tokenizer = create_tokenizer("mul_tokens")
        
        raw_text = "Calculate 6*9 and 3*4."
        injected = injector.inject(raw_text)
        tokens = tokenizer.encode(injected)
        decoded = tokenizer.decode(tokens)
        
        # Should match the injected form
        assert decoded == injected
        assert "<MUL_6_9_54>" in decoded
        assert "<MUL_3_4_12>" in decoded
    
    def test_baseline_no_injection_benefit(self):
        """Baseline tokenizer doesn't benefit from injection (still multi-token)."""
        injector = create_injector("weak")
        baseline = create_tokenizer("baseline")
        mul_tok = create_tokenizer("mul_tokens")
        
        raw_text = "6*9"
        injected = injector.inject(raw_text)
        
        baseline_tokens = baseline.encode(injected)
        mul_tokens = mul_tok.encode(injected)
        
        # Baseline: <MUL_6_9_54> is tokenized as multiple BPE tokens
        assert len(baseline_tokens) > 1
        
        # Mul-tokens: <MUL_6_9_54> is a single token
        assert len(mul_tokens) == 1
    
    def test_token_savings(self):
        """Mul-token condition should use fewer tokens."""
        injector = create_injector("weak")
        baseline = create_tokenizer("baseline")
        mul_tok = create_tokenizer("mul_tokens")
        
        raw_text = "The answers are 6*9, 3*4, and 7*8."
        injected = injector.inject(raw_text)
        
        baseline_tokens = baseline.encode(injected)
        mul_tokens = mul_tok.encode(injected)
        
        # Mul-token version should have fewer tokens
        assert len(mul_tokens) < len(baseline_tokens)
        
        # Specifically, we save tokens for each mul-fact
        # Each <MUL_x_y_z> in baseline is ~10 tokens, in mul is 1 token


class TestCreateInjector:
    """Tests for create_injector factory function."""
    
    def test_weak_mode(self):
        """'weak' mode should not require result."""
        injector = create_injector("weak")
        assert injector.require_result is False
    
    def test_strict_mode(self):
        """'strict' mode should require result."""
        injector = create_injector("strict")
        assert injector.require_result is True
    
    def test_invalid_mode(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError):
            create_injector("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

