#!/usr/bin/env python3
"""
Reward functions for GSM8K RL training.

Implements exact-match reward based on final numeric answer extraction.
GSM8K answers are formatted with "#### {number}" at the end.
"""

import re
from typing import List, Optional, Tuple, Union


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from a GSM8K-style response.
    
    Looks for patterns like:
    - "#### 42"
    - "#### -42" 
    - "#### 3.14"
    - "The answer is 42"
    
    Returns the extracted number as a string, or None if not found.
    """
    text = text.strip()
    
    # Primary pattern: #### followed by number
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?(?:,\d{3})*)', text)
    if match:
        # Remove commas from numbers like "1,234"
        return match.group(1).replace(',', '')
    
    # Fallback: "The answer is X" pattern
    match = re.search(r'[Tt]he answer is\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)
    
    # Fallback: last number in the text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    
    - Strips whitespace
    - Removes commas
    - Converts to standard numeric format
    """
    if answer is None:
        return ""
    
    answer = str(answer).strip()
    answer = answer.replace(',', '')
    
    # Try to normalize to float then back to string
    try:
        num = float(answer)
        # Handle infinity and very large floats
        if not (-1e15 < num < 1e15):
            return answer  # Return original for out-of-range values
        # If it's an integer, format without decimal
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        return answer


def compute_exact_match_reward(
    prediction: str,
    ground_truth: str,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
) -> Tuple[float, dict]:
    """
    Compute exact match reward for a single prediction.
    
    Args:
        prediction: Model's generated response
        ground_truth: Expected answer (can be full response or just the number)
        correct_reward: Reward for correct answer
        incorrect_reward: Reward for incorrect answer
        
    Returns:
        (reward, info_dict) where info_dict contains extraction details
    """
    # Extract answers
    pred_answer = extract_answer(prediction)
    true_answer = extract_answer(ground_truth)
    
    # If ground truth doesn't have #### pattern, treat it as raw answer
    if true_answer is None:
        true_answer = normalize_answer(ground_truth)
    else:
        true_answer = normalize_answer(true_answer)
    
    pred_answer = normalize_answer(pred_answer) if pred_answer else ""
    
    # Compare
    is_correct = (pred_answer == true_answer) and pred_answer != ""
    reward = correct_reward if is_correct else incorrect_reward
    
    info = {
        "pred_answer": pred_answer,
        "true_answer": true_answer,
        "is_correct": is_correct,
        "reward": reward,
    }
    
    return reward, info


def compute_batch_rewards(
    predictions: List[str],
    ground_truths: List[str],
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
) -> Tuple[List[float], List[dict]]:
    """
    Compute rewards for a batch of predictions.
    
    Args:
        predictions: List of model responses
        ground_truths: List of expected answers
        correct_reward: Reward for correct answer
        incorrect_reward: Reward for incorrect answer
        
    Returns:
        (rewards_list, info_list)
    """
    rewards = []
    infos = []
    
    for pred, truth in zip(predictions, ground_truths):
        reward, info = compute_exact_match_reward(
            pred, truth, correct_reward, incorrect_reward
        )
        rewards.append(reward)
        infos.append(info)
    
    return rewards, infos


class GSM8KRewardFunction:
    """
    Reward function class for use with TRL/GRPO.
    
    Computes exact-match reward based on final answer extraction.
    """
    
    def __init__(
        self,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
    ):
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.stats = {"correct": 0, "incorrect": 0, "total": 0}
    
    def __call__(
        self,
        responses: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        """
        Compute rewards for responses.
        
        Args:
            responses: Generated responses
            ground_truths: Expected answers
            
        Returns:
            List of rewards
        """
        rewards, infos = compute_batch_rewards(
            responses, ground_truths,
            self.correct_reward, self.incorrect_reward
        )
        
        # Update stats
        for info in infos:
            self.stats["total"] += 1
            if info["is_correct"]:
                self.stats["correct"] += 1
            else:
                self.stats["incorrect"] += 1
        
        return rewards
    
    @property
    def accuracy(self) -> float:
        """Current accuracy based on tracked stats."""
        if self.stats["total"] == 0:
            return 0.0
        return self.stats["correct"] / self.stats["total"]
    
    def reset_stats(self):
        """Reset tracking stats."""
        self.stats = {"correct": 0, "incorrect": 0, "total": 0}


if __name__ == "__main__":
    # Demo / test
    print("Testing reward extraction...")
    
    test_cases = [
        ("The answer is #### 42", "42"),
        ("Let me calculate... #### 123", "123"),
        ("The total is 50 + 7 = 57. #### 57", "57"),
        ("The answer is -5", "-5"),
        ("No answer here", ""),
        ("#### 1,234", "1234"),
    ]
    
    for text, expected in test_cases:
        extracted = extract_answer(text)
        normalized = normalize_answer(extracted) if extracted else ""
        status = "✓" if normalized == expected else "✗"
        print(f"  {status} '{text[:40]}...' -> '{normalized}' (expected: '{expected}')")
    
    print("\nTesting reward function...")
    reward_fn = GSM8KRewardFunction()
    
    responses = [
        "Let's solve this step by step. First... #### 42",
        "The calculation gives us #### 100",
        "I think it's about 50",
    ]
    truths = ["42", "99", "50"]
    
    rewards = reward_fn(responses, truths)
    print(f"  Rewards: {rewards}")
    print(f"  Accuracy: {reward_fn.accuracy:.2%}")

