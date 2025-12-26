"""Tokenizer module for domain-specific token injection experiments."""

from tokenizer.mul_facts import MulFactTokens, get_mul_token, get_mul_token_range

# Lazy imports for modules that may not be complete yet
__all__ = [
    "MulFactTokens",
    "get_mul_token",
    "get_mul_token_range",
]

# Conditional imports (these are added as modules are implemented)
try:
    from tokenizer.gpt2_tiktoken import GPT2TokenizerWithMulFacts, create_tokenizer
    __all__.extend(["GPT2TokenizerWithMulFacts", "create_tokenizer"])
except ImportError:
    pass

try:
    from tokenizer.inject_mul import MulExpressionInjector
    __all__.append("MulExpressionInjector")
except ImportError:
    pass

