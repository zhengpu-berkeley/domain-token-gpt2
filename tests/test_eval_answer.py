"""
Unit tests for answer extraction and reward computation.

Tests:
- extract_answer: Parse "#### N" pattern from various formats
- normalize_answer: Standardize numbers for comparison
- compute_exact_match_reward: Reward function correctness
"""

import pytest
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.rewards import (
    extract_answer,
    normalize_answer,
    compute_exact_match_reward,
    compute_batch_rewards,
    GSM8KRewardFunction,
)


class TestExtractAnswer:
    """Tests for extract_answer function."""

    def test_standard_format(self):
        """Should extract from standard #### N format."""
        assert extract_answer("#### 42") == "42"
        assert extract_answer("Some text #### 123") == "123"
        assert extract_answer("#### 0") == "0"

    def test_with_calculations(self):
        """Should extract from responses with step-by-step work."""
        text = "First, 5 + 3 = 8. Then 8 * 2 = 16. #### 16"
        assert extract_answer(text) == "16"

    def test_negative_numbers(self):
        """Should handle negative numbers."""
        assert extract_answer("#### -5") == "-5"
        assert extract_answer("The result is #### -123") == "-123"

    def test_decimal_numbers(self):
        """Should handle decimal numbers."""
        assert extract_answer("#### 3.14") == "3.14"
        assert extract_answer("#### 0.5") == "0.5"

    def test_numbers_with_commas(self):
        """Should handle numbers with thousand separators."""
        assert extract_answer("#### 1,234") == "1234"
        assert extract_answer("#### 1,234,567") == "1234567"

    def test_no_space_after_hashes(self):
        """Should handle missing space after ####."""
        assert extract_answer("####42") == "42"
        assert extract_answer("####-5") == "-5"

    def test_extra_whitespace(self):
        """Should handle extra whitespace."""
        assert extract_answer("  #### 42  ") == "42"
        assert extract_answer("####   42") == "42"

    def test_fallback_the_answer_is(self):
        """Should use 'The answer is X' as fallback."""
        assert extract_answer("The answer is 42") == "42"
        assert extract_answer("the answer is 100") == "100"

    def test_fallback_last_number(self):
        """Should use last number as final fallback."""
        assert extract_answer("The values are 10, 20, 30") == "30"
        assert extract_answer("Result: 99") == "99"

    def test_no_answer(self):
        """Should return None when no number found."""
        assert extract_answer("No numbers here") is None
        assert extract_answer("") is None

    def test_prefers_hash_pattern(self):
        """Should prefer #### pattern over fallbacks."""
        # The #### pattern should take precedence
        text = "The answer is 50. #### 42"
        assert extract_answer(text) == "42"


class TestNormalizeAnswer:
    """Tests for normalize_answer function."""

    def test_integer_normalization(self):
        """Should normalize integers correctly."""
        assert normalize_answer("42") == "42"
        assert normalize_answer("42.0") == "42"
        assert normalize_answer("  42  ") == "42"

    def test_decimal_normalization(self):
        """Should preserve meaningful decimals."""
        assert normalize_answer("3.14") == "3.14"
        assert normalize_answer("0.5") == "0.5"

    def test_comma_removal(self):
        """Should remove thousand separators."""
        assert normalize_answer("1,234") == "1234"
        assert normalize_answer("1,234,567") == "1234567"

    def test_negative_numbers(self):
        """Should handle negative numbers."""
        assert normalize_answer("-5") == "-5"
        assert normalize_answer("-3.14") == "-3.14"

    def test_none_handling(self):
        """Should handle None input."""
        assert normalize_answer(None) == ""

    def test_non_numeric(self):
        """Should return original for non-numeric strings."""
        assert normalize_answer("hello") == "hello"


class TestComputeExactMatchReward:
    """Tests for compute_exact_match_reward function."""

    def test_correct_answer(self):
        """Should return correct_reward for matching answers."""
        reward, info = compute_exact_match_reward(
            "The answer is #### 42",
            "42",
        )
        assert reward == 1.0
        assert info["is_correct"] is True
        assert info["pred_answer"] == "42"
        assert info["true_answer"] == "42"

    def test_incorrect_answer(self):
        """Should return incorrect_reward for non-matching answers."""
        reward, info = compute_exact_match_reward(
            "The answer is #### 42",
            "100",
        )
        assert reward == 0.0
        assert info["is_correct"] is False

    def test_custom_rewards(self):
        """Should use custom reward values."""
        reward, _ = compute_exact_match_reward(
            "#### 42",
            "42",
            correct_reward=10.0,
            incorrect_reward=-1.0,
        )
        assert reward == 10.0

        reward, _ = compute_exact_match_reward(
            "#### 99",
            "42",
            correct_reward=10.0,
            incorrect_reward=-1.0,
        )
        assert reward == -1.0

    def test_no_prediction_answer(self):
        """Should return incorrect for missing prediction."""
        reward, info = compute_exact_match_reward(
            "I don't know",
            "42",
        )
        # Last number fallback will find nothing, so incorrect
        assert reward == 0.0


class TestComputeBatchRewards:
    """Tests for compute_batch_rewards function."""

    def test_batch_processing(self):
        """Should process multiple predictions correctly."""
        predictions = [
            "#### 42",
            "#### 100",
            "#### 50",
        ]
        ground_truths = ["42", "99", "50"]

        rewards, infos = compute_batch_rewards(predictions, ground_truths)

        assert len(rewards) == 3
        assert rewards[0] == 1.0  # Correct
        assert rewards[1] == 0.0  # Wrong
        assert rewards[2] == 1.0  # Correct

        assert infos[0]["is_correct"] is True
        assert infos[1]["is_correct"] is False
        assert infos[2]["is_correct"] is True


class TestGSM8KRewardFunction:
    """Tests for GSM8KRewardFunction class."""

    def test_callable(self):
        """Should work as a callable."""
        reward_fn = GSM8KRewardFunction()
        rewards = reward_fn(
            ["#### 42", "#### 100"],
            ["42", "99"],
        )
        assert rewards == [1.0, 0.0]

    def test_accuracy_tracking(self):
        """Should track accuracy correctly."""
        reward_fn = GSM8KRewardFunction()

        reward_fn(["#### 42", "#### 50", "#### 100"], ["42", "50", "99"])

        assert reward_fn.stats["correct"] == 2
        assert reward_fn.stats["incorrect"] == 1
        assert reward_fn.stats["total"] == 3
        assert reward_fn.accuracy == pytest.approx(2 / 3)

    def test_reset_stats(self):
        """Should reset stats correctly."""
        reward_fn = GSM8KRewardFunction()
        reward_fn(["#### 42"], ["42"])
        assert reward_fn.stats["total"] == 1

        reward_fn.reset_stats()
        assert reward_fn.stats["total"] == 0
        assert reward_fn.accuracy == 0.0


class TestAnswerExtractionEdgeCases:
    """Additional edge case tests using fixtures."""

    def test_all_extraction_cases(self, answer_extraction_cases):
        """Test all answer extraction cases from fixture."""
        for text, expected in answer_extraction_cases:
            extracted = extract_answer(text)
            if expected is None:
                assert extracted is None, f"Expected None for '{text}', got '{extracted}'"
            else:
                normalized = normalize_answer(extracted)
                assert normalized == expected, f"For '{text}': expected '{expected}', got '{normalized}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

