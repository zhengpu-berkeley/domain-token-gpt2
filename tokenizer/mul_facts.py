"""
Multiplication-fact token generation and ID mapping.

Generates structured tokens like <MUL_6_9_54> for multiplication facts.
Handles canonicalization (a <= b) for commutative equivalence.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# GPT-2 base vocab size is 50257. We round up to 50304 for efficiency (divisible by 64).
# Our mul tokens will be appended after that.
GPT2_BASE_VOCAB_SIZE = 50257
GPT2_PADDED_VOCAB_SIZE = 50304  # Karpathy's default for efficiency

# Range for multiplication table (1-9 gives 81 tokens)
MUL_RANGE_MIN = 1
MUL_RANGE_MAX = 9


@dataclass(frozen=True)
class MulToken:
    """Represents a single multiplication fact token."""
    a: int  # First operand (canonical: a <= b)
    b: int  # Second operand
    c: int  # Product (a * b)
    
    @property
    def token_str(self) -> str:
        """Returns the token string like <MUL_6_9_54>."""
        return f"<MUL_{self.a}_{self.b}_{self.c}>"
    
    def __str__(self) -> str:
        return self.token_str


def canonicalize(a: int, b: int) -> Tuple[int, int]:
    """Canonicalize operand order for commutativity: ensure a <= b."""
    return (a, b) if a <= b else (b, a)


def get_mul_token(a: int, b: int) -> Optional[MulToken]:
    """
    Get the MulToken for a multiplication fact.
    
    Args:
        a: First operand
        b: Second operand
        
    Returns:
        MulToken if a,b are in range, None otherwise.
    """
    if not (MUL_RANGE_MIN <= a <= MUL_RANGE_MAX and 
            MUL_RANGE_MIN <= b <= MUL_RANGE_MAX):
        return None
    
    a_canon, b_canon = canonicalize(a, b)
    return MulToken(a=a_canon, b=b_canon, c=a_canon * b_canon)


def get_mul_token_range() -> Tuple[int, int]:
    """Returns (min, max) inclusive range for multiplication operands."""
    return (MUL_RANGE_MIN, MUL_RANGE_MAX)


class MulFactTokens:
    """
    Manages the complete set of multiplication fact tokens.
    
    Generates all 81 tokens for 1x1 through 9x9 (canonicalized),
    and assigns them sequential token IDs starting after GPT-2's vocab.
    """
    
    def __init__(self, base_vocab_size: int = GPT2_PADDED_VOCAB_SIZE):
        """
        Initialize the multiplication token set.
        
        Args:
            base_vocab_size: Starting token ID for mul tokens (typically 50304).
        """
        self.base_vocab_size = base_vocab_size
        self._tokens: List[MulToken] = []
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, MulToken] = {}
        self._fact_to_token: Dict[Tuple[int, int], MulToken] = {}
        
        self._generate_tokens()
    
    def _generate_tokens(self) -> None:
        """Generate all multiplication fact tokens with IDs."""
        token_id = self.base_vocab_size
        
        # Generate canonicalized tokens (a <= b to avoid duplicates)
        for a in range(MUL_RANGE_MIN, MUL_RANGE_MAX + 1):
            for b in range(a, MUL_RANGE_MAX + 1):  # b >= a for canonical form
                token = MulToken(a=a, b=b, c=a * b)
                self._tokens.append(token)
                self._token_to_id[token.token_str] = token_id
                self._id_to_token[token_id] = token
                self._fact_to_token[(a, b)] = token
                token_id += 1
    
    @property
    def num_tokens(self) -> int:
        """Number of multiplication tokens."""
        return len(self._tokens)
    
    @property
    def vocab_size(self) -> int:
        """Total vocab size including base + mul tokens."""
        return self.base_vocab_size + self.num_tokens
    
    @property
    def token_id_range(self) -> Tuple[int, int]:
        """Returns (first_id, last_id) inclusive range for mul token IDs."""
        return (self.base_vocab_size, self.base_vocab_size + self.num_tokens - 1)
    
    @property
    def all_tokens(self) -> List[MulToken]:
        """List of all MulToken objects."""
        return self._tokens.copy()
    
    @property
    def all_token_strings(self) -> List[str]:
        """List of all token strings like ['<MUL_1_1_1>', '<MUL_1_2_2>', ...]."""
        return [t.token_str for t in self._tokens]
    
    def get_token_id(self, token_str: str) -> Optional[int]:
        """Get token ID for a token string, or None if not found."""
        return self._token_to_id.get(token_str)
    
    def get_token_by_id(self, token_id: int) -> Optional[MulToken]:
        """Get MulToken for a token ID, or None if not found."""
        return self._id_to_token.get(token_id)
    
    def get_token_for_fact(self, a: int, b: int) -> Optional[MulToken]:
        """
        Get the MulToken for a multiplication fact.
        
        Automatically canonicalizes (a,b) to (min,max) order.
        Returns None if either operand is out of range.
        """
        if not (MUL_RANGE_MIN <= a <= MUL_RANGE_MAX and 
                MUL_RANGE_MIN <= b <= MUL_RANGE_MAX):
            return None
        
        a_canon, b_canon = canonicalize(a, b)
        return self._fact_to_token.get((a_canon, b_canon))
    
    def get_token_id_for_fact(self, a: int, b: int) -> Optional[int]:
        """Get token ID for a multiplication fact, or None if out of range."""
        token = self.get_token_for_fact(a, b)
        if token is None:
            return None
        return self._token_to_id.get(token.token_str)
    
    def is_mul_token_id(self, token_id: int) -> bool:
        """Check if a token ID is a multiplication token."""
        first, last = self.token_id_range
        return first <= token_id <= last
    
    def get_special_tokens_dict(self) -> Dict[str, int]:
        """
        Returns a dict mapping token strings to IDs.
        Useful for adding to tokenizer.
        """
        return self._token_to_id.copy()
    
    def summary(self) -> str:
        """Human-readable summary of the token set."""
        first_id, last_id = self.token_id_range
        return (
            f"MulFactTokens:\n"
            f"  Range: {MUL_RANGE_MIN}x{MUL_RANGE_MIN} to "
            f"{MUL_RANGE_MAX}x{MUL_RANGE_MAX}\n"
            f"  Num tokens: {self.num_tokens}\n"
            f"  Token ID range: [{first_id}, {last_id}]\n"
            f"  Total vocab size: {self.vocab_size}\n"
            f"  First token: {self._tokens[0]}\n"
            f"  Last token: {self._tokens[-1]}"
        )


# Singleton instance for convenience
_default_instance: Optional[MulFactTokens] = None


def get_default_mul_tokens() -> MulFactTokens:
    """Get the default MulFactTokens instance (singleton)."""
    global _default_instance
    if _default_instance is None:
        _default_instance = MulFactTokens()
    return _default_instance


if __name__ == "__main__":
    # Demo / quick test
    mul_tokens = MulFactTokens()
    print(mul_tokens.summary())
    print()
    
    # Test canonicalization
    print("Testing canonicalization:")
    for (a, b) in [(6, 9), (9, 6), (3, 3), (1, 9)]:
        token = mul_tokens.get_token_for_fact(a, b)
        token_id = mul_tokens.get_token_id_for_fact(a, b)
        print(f"  {a}x{b} -> {token} (ID: {token_id})")
    
    print()
    print("All tokens:")
    for token in mul_tokens.all_tokens[:5]:
        token_id = mul_tokens.get_token_id(token.token_str)
        print(f"  {token.token_str} -> ID {token_id}")
    print("  ...")
    for token in mul_tokens.all_tokens[-3:]:
        token_id = mul_tokens.get_token_id(token.token_str)
        print(f"  {token.token_str} -> ID {token_id}")

