"""
Shared test fixtures for domain-token-gpt2 tests.

These fixtures provide small, reusable test data that doesn't require
downloading large datasets or models.
"""

import pytest
import torch
from dataclasses import dataclass


@dataclass
class TinyModelConfig:
    """Minimal GPT config for testing (tiny model, fast tests)."""
    vocab_size: int = 50349
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 128


@pytest.fixture
def tiny_config():
    """Return a minimal GPT config for testing."""
    return TinyModelConfig()


@pytest.fixture
def device():
    """Return best available device (prefer CPU for reproducibility in tests)."""
    return torch.device("cpu")


@pytest.fixture
def sample_gsm8k_examples():
    """Sample GSM8K-style examples for testing."""
    return [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = <<9*2=18>>$18 every day at the farmer's market.\n#### 18",
            "numeric_answer": "18",
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "answer": "It takes 2/2 = <<2/2=1>>1 bolt of white fiber.\nSo the total amount of fabric is 2 + 1 = <<2+1=3>>3 bolts.\n#### 3",
            "numeric_answer": "3",
        },
    ]


@pytest.fixture
def sample_tinygsm_examples():
    """Sample TinyGSM-style examples for testing."""
    return [
        {
            "question": "Sarah has 5 baskets with 8 apples each. How many apples does she have?",
            "solution": "Sarah has 5 baskets with 8 apples each.\nTotal apples = 5 × 8 = 40 apples.\n#### 40",
            "numeric_answer": "40",
        },
        {
            "question": "Tom bought 3 books at $7 each. How much did he spend?",
            "solution": "Tom bought 3 books at $7 each.\nTotal cost = 3 × 7 = $21.\n#### 21",
            "numeric_answer": "21",
        },
    ]


@pytest.fixture
def sample_mul_expressions():
    """Sample multiplication expressions for testing injection."""
    return [
        ("6*9", "<MUL_6_9_54>"),
        ("7 × 8", "<MUL_7_8_56>"),
        ("3x4", "<MUL_3_4_12>"),
        ("9*6=54", "<MUL_6_9_54>"),  # Canonicalized
        ("10*5", "10*5"),  # Out of range, not injected
    ]


@pytest.fixture
def answer_extraction_cases():
    """Test cases for answer extraction from model outputs."""
    return [
        # (input_text, expected_extracted_answer)
        ("The answer is #### 42", "42"),
        ("Let me calculate... #### 123", "123"),
        ("The total is 50 + 7 = 57. #### 57", "57"),
        ("#### -5", "-5"),
        ("#### 3.14", "3.14"),
        ("#### 1,234", "1234"),  # Commas removed
        ("The answer is 42", "42"),  # Fallback pattern
        ("No answer marker, just 99", "99"),  # Last number fallback
        ("No numbers at all", None),
        # Edge cases
        ("#### 0", "0"),
        ("####42", "42"),  # No space
        ("  ####   42  ", "42"),  # Extra whitespace
    ]


@pytest.fixture
def tiny_nanogpt_checkpoint(tiny_config):
    """Create a minimal nanoGPT-style checkpoint for testing."""
    config = tiny_config
    
    # Create random weights for a tiny model
    state_dict = {}
    
    # Embeddings
    state_dict["transformer.wte.weight"] = torch.randn(config.vocab_size, config.n_embd)
    state_dict["transformer.wpe.weight"] = torch.randn(config.block_size, config.n_embd)
    
    # Final layer norm
    state_dict["transformer.ln_f.weight"] = torch.ones(config.n_embd)
    state_dict["transformer.ln_f.bias"] = torch.zeros(config.n_embd)
    
    # LM head (tied with wte in some implementations)
    state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"].clone()
    
    # Layers
    for i in range(config.n_layer):
        prefix = f"transformer.h.{i}"
        
        # Attention
        state_dict[f"{prefix}.ln_1.weight"] = torch.ones(config.n_embd)
        state_dict[f"{prefix}.ln_1.bias"] = torch.zeros(config.n_embd)
        state_dict[f"{prefix}.attn.c_attn.weight"] = torch.randn(3 * config.n_embd, config.n_embd)
        state_dict[f"{prefix}.attn.c_attn.bias"] = torch.zeros(3 * config.n_embd)
        state_dict[f"{prefix}.attn.c_proj.weight"] = torch.randn(config.n_embd, config.n_embd)
        state_dict[f"{prefix}.attn.c_proj.bias"] = torch.zeros(config.n_embd)
        
        # MLP
        state_dict[f"{prefix}.ln_2.weight"] = torch.ones(config.n_embd)
        state_dict[f"{prefix}.ln_2.bias"] = torch.zeros(config.n_embd)
        state_dict[f"{prefix}.mlp.c_fc.weight"] = torch.randn(4 * config.n_embd, config.n_embd)
        state_dict[f"{prefix}.mlp.c_fc.bias"] = torch.zeros(4 * config.n_embd)
        state_dict[f"{prefix}.mlp.c_proj.weight"] = torch.randn(config.n_embd, 4 * config.n_embd)
        state_dict[f"{prefix}.mlp.c_proj.bias"] = torch.zeros(config.n_embd)
    
    checkpoint = {
        "model": state_dict,
        "config": {
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
        },
        "step": 100,
        "val_loss": 5.0,
    }
    
    return checkpoint

