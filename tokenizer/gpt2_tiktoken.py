"""
GPT-2 tokenizer wrapper using tiktoken with optional mul-fact special tokens.

Provides a unified interface for both baseline and mul-token conditions:
- Baseline: standard GPT-2 BPE encoding
- Mul-tokens: GPT-2 BPE + special tokens for multiplication facts

Both conditions use the SAME vocab_size (with reserved token IDs) to ensure
identical model architecture.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import tiktoken

from tokenizer.mul_facts import MulFactTokens, get_default_mul_tokens


class GPT2TokenizerWithMulFacts:
    """
    GPT-2 tokenizer with optional multiplication fact tokens.
    
    Key design: both conditions share the same vocab_size.
    - Baseline condition: mul token IDs are reserved but never produced
    - Mul-token condition: mul token IDs are produced when special tokens appear
    
    This ensures identical model architecture across conditions.
    """
    
    def __init__(
        self,
        enable_mul_tokens: bool = True,
        mul_tokens: Optional[MulFactTokens] = None,
    ):
        """
        Initialize the tokenizer.
        
        Args:
            enable_mul_tokens: If True, recognize and encode mul-fact tokens.
                               If False, treat them as regular text (baseline).
            mul_tokens: MulFactTokens instance. Uses default if None.
        """
        self.enable_mul_tokens = enable_mul_tokens
        self.mul_tokens = mul_tokens or get_default_mul_tokens()
        
        # Load base GPT-2 encoding
        self._base_enc = tiktoken.get_encoding("gpt2")
        
        # Build special token pattern for mul tokens
        if self.enable_mul_tokens:
            self._special_token_pattern = self._build_special_token_pattern()
        else:
            self._special_token_pattern = None
        
        # Cache vocab size
        self._vocab_size = self.mul_tokens.vocab_size
    
    def _build_special_token_pattern(self) -> re.Pattern:
        """Build regex pattern to match mul-fact tokens."""
        # Pattern matches <MUL_a_b_c> where a,b,c are digits
        # We use a precise pattern to avoid false matches
        return re.compile(r"<MUL_(\d+)_(\d+)_(\d+)>")
    
    @property
    def vocab_size(self) -> int:
        """
        Total vocabulary size (base GPT-2 + reserved mul tokens).
        
        This is the same for both conditions to ensure identical embedding size.
        """
        return self._vocab_size
    
    @property
    def base_vocab_size(self) -> int:
        """Base GPT-2 vocabulary size (before mul tokens)."""
        return self.mul_tokens.base_vocab_size
    
    @property
    def eot_token(self) -> int:
        """End-of-text token ID."""
        return self._base_enc.eot_token
    
    @property
    def eot_token_str(self) -> str:
        """End-of-text token string."""
        return "<|endoftext|>"
    
    def encode(
        self,
        text: str,
        allowed_special: Union[set, str] = "none",
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            allowed_special: Set of allowed special tokens, or "none"/"all"
            
        Returns:
            List of token IDs
        """
        if not self.enable_mul_tokens or self._special_token_pattern is None:
            # Baseline condition: use standard GPT-2 encoding
            if allowed_special == "none":
                return self._base_enc.encode(text, allowed_special=set())
            elif allowed_special == "all":
                return self._base_enc.encode(text, allowed_special="all")
            else:
                return self._base_enc.encode(text, allowed_special=allowed_special)
        
        # Mul-token condition: handle special tokens
        return self._encode_with_mul_tokens(text, allowed_special)
    
    def _encode_with_mul_tokens(
        self,
        text: str,
        allowed_special: Union[set, str],
    ) -> List[int]:
        """Encode text with mul-token handling."""
        tokens = []
        last_end = 0
        
        # Find all mul-token matches
        for match in self._special_token_pattern.finditer(text):
            start, end = match.span()
            
            # Encode text before this match
            if start > last_end:
                prefix = text[last_end:start]
                if allowed_special == "none":
                    tokens.extend(self._base_enc.encode(prefix, allowed_special=set()))
                elif allowed_special == "all":
                    tokens.extend(self._base_enc.encode(prefix, allowed_special="all"))
                else:
                    tokens.extend(self._base_enc.encode(prefix, allowed_special=allowed_special))
            
            # Parse and encode the mul token
            a, b, c = int(match.group(1)), int(match.group(2)), int(match.group(3))
            token_id = self.mul_tokens.get_token_id_for_fact(a, b)
            
            if token_id is not None and a * b == c:
                # Valid mul token
                tokens.append(token_id)
            else:
                # Invalid or out-of-range: encode as regular text
                if allowed_special == "none":
                    tokens.extend(self._base_enc.encode(match.group(0), allowed_special=set()))
                elif allowed_special == "all":
                    tokens.extend(self._base_enc.encode(match.group(0), allowed_special="all"))
                else:
                    tokens.extend(self._base_enc.encode(match.group(0), allowed_special=allowed_special))
            
            last_end = end
        
        # Encode remaining text
        if last_end < len(text):
            suffix = text[last_end:]
            if allowed_special == "none":
                tokens.extend(self._base_enc.encode(suffix, allowed_special=set()))
            elif allowed_special == "all":
                tokens.extend(self._base_enc.encode(suffix, allowed_special="all"))
            else:
                tokens.extend(self._base_enc.encode(suffix, allowed_special=allowed_special))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text string
        """
        result = []
        base_tokens = []
        
        for token_id in tokens:
            if self.mul_tokens.is_mul_token_id(token_id):
                # Flush base tokens first
                if base_tokens:
                    result.append(self._base_enc.decode(base_tokens))
                    base_tokens = []
                
                # Decode mul token
                mul_token = self.mul_tokens.get_token_by_id(token_id)
                if mul_token:
                    result.append(mul_token.token_str)
                else:
                    # Unknown mul token ID (shouldn't happen)
                    result.append(f"<UNK_{token_id}>")
            else:
                # Regular token
                base_tokens.append(token_id)
        
        # Flush remaining base tokens
        if base_tokens:
            result.append(self._base_enc.decode(base_tokens))
        
        return "".join(result)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """Decode a batch of token lists."""
        return [self.decode(tokens) for tokens in token_lists]
    
    def get_metadata(self) -> Dict:
        """
        Get tokenizer metadata for saving/logging.
        
        Returns dict with:
        - vocab_size
        - base_vocab_size
        - enable_mul_tokens
        - mul_token_range
        - num_mul_tokens
        """
        first_id, last_id = self.mul_tokens.token_id_range
        return {
            "vocab_size": self.vocab_size,
            "base_vocab_size": self.base_vocab_size,
            "enable_mul_tokens": self.enable_mul_tokens,
            "mul_token_id_start": first_id,
            "mul_token_id_end": last_id,
            "num_mul_tokens": self.mul_tokens.num_tokens,
            "mul_range_min": 1,
            "mul_range_max": 9,
        }
    
    def save_metadata(self, path: Union[str, Path]) -> None:
        """Save tokenizer metadata to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_metadata(), f, indent=2)
    
    @classmethod
    def load_metadata(cls, path: Union[str, Path]) -> Dict:
        """Load tokenizer metadata from JSON file."""
        with open(path, "r") as f:
            return json.load(f)
    
    def count_mul_tokens(self, tokens: List[int]) -> int:
        """Count how many mul-fact tokens are in a token list."""
        return sum(1 for t in tokens if self.mul_tokens.is_mul_token_id(t))
    
    def get_mul_token_stats(self, tokens: List[int]) -> Dict[str, int]:
        """
        Get statistics about mul-token usage in a token list.
        
        Returns:
            Dict with total_tokens, mul_tokens, regular_tokens, mul_token_ratio
        """
        mul_count = self.count_mul_tokens(tokens)
        total = len(tokens)
        return {
            "total_tokens": total,
            "mul_tokens": mul_count,
            "regular_tokens": total - mul_count,
            "mul_token_ratio": mul_count / total if total > 0 else 0.0,
        }


def create_tokenizer(condition: str = "mul_tokens") -> GPT2TokenizerWithMulFacts:
    """
    Factory function to create tokenizer for a condition.
    
    Args:
        condition: "baseline" or "mul_tokens"
        
    Returns:
        Configured tokenizer instance
    """
    if condition == "baseline":
        return GPT2TokenizerWithMulFacts(enable_mul_tokens=False)
    elif condition == "mul_tokens":
        return GPT2TokenizerWithMulFacts(enable_mul_tokens=True)
    else:
        raise ValueError(f"Unknown condition: {condition}. Use 'baseline' or 'mul_tokens'.")


if __name__ == "__main__":
    # Demo / quick test
    print("=" * 60)
    print("Testing GPT2TokenizerWithMulFacts")
    print("=" * 60)
    
    # Test both conditions
    for condition in ["baseline", "mul_tokens"]:
        print(f"\n--- Condition: {condition} ---")
        tokenizer = create_tokenizer(condition)
        
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Enable mul tokens: {tokenizer.enable_mul_tokens}")
        
        # Test encode/decode
        test_texts = [
            "Hello, world!",
            "6 times 9 equals <MUL_6_9_54>.",
            "What is <MUL_3_4_12>? The answer is 12.",
        ]
        
        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            stats = tokenizer.get_mul_token_stats(tokens)
            
            print(f"\n  Input:   '{text}'")
            print(f"  Tokens:  {tokens}")
            print(f"  Decoded: '{decoded}'")
            print(f"  Stats:   {stats}")
            
            # Verify roundtrip
            if text == decoded:
                print("  Roundtrip: OK")
            else:
                print(f"  Roundtrip: MISMATCH")
    
    print("\n" + "=" * 60)
    print("Metadata:")
    tokenizer = create_tokenizer("mul_tokens")
    for k, v in tokenizer.get_metadata().items():
        print(f"  {k}: {v}")

