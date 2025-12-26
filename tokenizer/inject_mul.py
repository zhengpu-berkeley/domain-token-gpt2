"""
Multiplication expression injector.

Preprocesses text to replace multiplication expressions (like `6*9`, `6 × 9`)
with structured mul-fact tokens (like `<MUL_6_9_54>`).

Supports canonicalization: both `6*9` and `9*6` become `<MUL_6_9_54>`.
"""

import re
from typing import List, Optional, Tuple

from tokenizer.mul_facts import (
    MulFactTokens,
    get_default_mul_tokens,
    get_mul_token_range,
    canonicalize,
)


class MulExpressionInjector:
    """
    Preprocessor that injects multiplication-fact tokens into text.
    
    Detects patterns like:
    - `6*9` or `6 * 9` (asterisk)
    - `6×9` or `6 × 9` (unicode multiplication sign)
    - `6x9` or `6 x 9` (lowercase x)
    
    And optionally detects patterns with results:
    - `6*9=54` or `6 * 9 = 54`
    
    Rewrites matched patterns into structured tokens like `<MUL_6_9_54>`.
    """
    
    # Patterns for multiplication operators
    MUL_OPERATORS = [
        r'\*',      # asterisk
        r'×',       # unicode multiplication sign
        r'x(?![a-z])',  # lowercase x (but not if followed by letter, to avoid "6x" as variable)
    ]
    
    def __init__(
        self,
        mul_tokens: Optional[MulFactTokens] = None,
        require_result: bool = False,
        verify_result: bool = True,
    ):
        """
        Initialize the injector.
        
        Args:
            mul_tokens: MulFactTokens instance. Uses default if None.
            require_result: If True, only inject when `=c` is present (strict mode).
                           If False, inject for bare `a*b` expressions (weak supervision).
            verify_result: If True, verify that `c` matches `a*b` when result is present.
        """
        self.mul_tokens = mul_tokens or get_default_mul_tokens()
        self.require_result = require_result
        self.verify_result = verify_result
        
        self._min_val, self._max_val = get_mul_token_range()
        self._pattern = self._build_pattern()
    
    def _build_pattern(self) -> re.Pattern:
        """Build regex pattern for multiplication expressions."""
        # Digit pattern for operands (1-2 digits to catch 1-9 and potentially 10-20)
        digit = r'(\d{1,2})'
        
        # Optional whitespace
        ws = r'\s*'
        
        # Multiplication operator alternatives
        mul_op = f'(?:{"|".join(self.MUL_OPERATORS)})'
        
        # Optional result part: = c
        result_part = rf'(?:{ws}={ws}(\d+))?'
        
        # Full pattern: a * b [= c]
        pattern = f'{digit}{ws}{mul_op}{ws}{digit}{result_part}'
        
        return re.compile(pattern, re.IGNORECASE)
    
    def _in_range(self, val: int) -> bool:
        """Check if value is in valid range for mul tokens."""
        return self._min_val <= val <= self._max_val
    
    def _should_inject(
        self,
        a: int,
        b: int,
        c: Optional[int],
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if we should inject a mul token for this expression.
        
        Returns:
            (should_inject, token_string or None)
        """
        # Check range
        if not (self._in_range(a) and self._in_range(b)):
            return False, None
        
        expected_c = a * b
        
        # If result is present, optionally verify it
        if c is not None:
            if self.verify_result and c != expected_c:
                return False, None
        else:
            # No result present
            if self.require_result:
                return False, None
        
        # Get token
        token = self.mul_tokens.get_token_for_fact(a, b)
        if token is None:
            return False, None
        
        return True, token.token_str
    
    def inject(self, text: str) -> str:
        """
        Process text and inject mul-fact tokens.
        
        Args:
            text: Input text
            
        Returns:
            Text with multiplication expressions replaced by mul tokens
        """
        def replace_fn(match: re.Match) -> str:
            a_str, b_str, c_str = match.group(1), match.group(2), match.group(3)
            a, b = int(a_str), int(b_str)
            c = int(c_str) if c_str is not None else None
            
            should_inject, token_str = self._should_inject(a, b, c)
            
            if should_inject and token_str:
                return token_str
            else:
                return match.group(0)  # Return original
        
        return self._pattern.sub(replace_fn, text)
    
    def inject_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts."""
        return [self.inject(text) for text in texts]
    
    def find_expressions(self, text: str) -> List[dict]:
        """
        Find all multiplication expressions in text (for debugging/analysis).
        
        Returns list of dicts with keys:
        - span: (start, end) positions
        - original: original text
        - a, b: operands
        - c: result (if present)
        - would_inject: whether this would be injected
        - token: resulting token string (if would_inject)
        """
        results = []
        for match in self._pattern.finditer(text):
            a_str, b_str, c_str = match.group(1), match.group(2), match.group(3)
            a, b = int(a_str), int(b_str)
            c = int(c_str) if c_str is not None else None
            
            should_inject, token_str = self._should_inject(a, b, c)
            
            results.append({
                "span": match.span(),
                "original": match.group(0),
                "a": a,
                "b": b,
                "c": c,
                "would_inject": should_inject,
                "token": token_str if should_inject else None,
            })
        
        return results


def create_injector(
    mode: str = "weak",
    mul_tokens: Optional[MulFactTokens] = None,
) -> MulExpressionInjector:
    """
    Factory function to create injector with preset modes.
    
    Args:
        mode: "weak" (inject bare a*b) or "strict" (require =c)
        mul_tokens: Optional MulFactTokens instance
        
    Returns:
        Configured MulExpressionInjector
    """
    if mode == "weak":
        return MulExpressionInjector(
            mul_tokens=mul_tokens,
            require_result=False,
            verify_result=True,
        )
    elif mode == "strict":
        return MulExpressionInjector(
            mul_tokens=mul_tokens,
            require_result=True,
            verify_result=True,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'weak' or 'strict'.")


if __name__ == "__main__":
    # Demo / quick test
    print("=" * 60)
    print("Testing MulExpressionInjector")
    print("=" * 60)
    
    # Test both modes
    for mode in ["weak", "strict"]:
        print(f"\n--- Mode: {mode} ---")
        injector = create_injector(mode)
        
        test_texts = [
            "What is 6*9?",
            "Calculate 6 * 9 = 54.",
            "The answer to 9×6 is 54.",
            "Try 3x4 and 4 x 5.",
            "6*9=54 is a multiplication fact.",
            "But 6*9=55 is wrong.",  # Wrong result
            "Large numbers: 15*20=300",  # Out of range
            "Variables like 2x are not touched.",
        ]
        
        for text in test_texts:
            result = injector.inject(text)
            changed = text != result
            marker = ">>>" if changed else "   "
            print(f"  {marker} '{text}'")
            if changed:
                print(f"       -> '{result}'")
    
    print("\n" + "=" * 60)
    print("Expression analysis (weak mode):")
    print("=" * 60)
    
    injector = create_injector("weak")
    sample = "Calculate 6*9, then 3 × 4 = 12, and finally 10*10."
    print(f"\nText: '{sample}'")
    print("Expressions found:")
    for expr in injector.find_expressions(sample):
        print(f"  - '{expr['original']}' at {expr['span']}")
        print(f"    a={expr['a']}, b={expr['b']}, c={expr['c']}")
        print(f"    would_inject={expr['would_inject']}, token={expr['token']}")

