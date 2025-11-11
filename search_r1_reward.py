#!/usr/bin/env python3
"""
Reward function for Search-R1 verl training.
Computes rewards based on answer quality and search effectiveness.
"""

import re
from typing import Dict, Any, List, Union
from utils import calculate_metrics, extract_answer


def compute_score(response: str, golden_answers: List[str], method: str = "strict",
                 format_score: float = 0.1, score: float = 1.0) -> float:
    """
    The scoring function for Search-R1.

    Args:
        response: the response text
        golden_answers: list of ground truth answers
        method: the method to extract the solution ('strict' or 'flexible')
        format_score: the score for format compliance
        score: the score for correct answer
    """
    if not isinstance(response, str):
        return 0.0

    if not golden_answers:
        return 0.0

    # Extract answer from response
    extracted_answer = extract_answer(response)

    if not extracted_answer:
        # No answer provided - small penalty
        return -0.1

    # Compute answer quality metrics
    metrics = calculate_metrics(extracted_answer, golden_answers)

    # Base reward from F1 score (0-1 scale)
    base_reward = metrics['f1'] * score

    # Bonus for exact match
    if metrics['em'] > 0:
        base_reward += 0.2

    # Bonus for good cover match
    if metrics['cover_match'] > 0.8:
        base_reward += 0.1

    # Check for proper reasoning format
    format_bonus = _check_reasoning_format(response)
    base_reward += format_bonus

    # Check for effective search usage
    search_bonus = _check_search_effectiveness(response)
    base_reward += search_bonus

    # Penalty for malformed responses
    if _is_malformed_response(response):
        base_reward -= 0.2

    return max(0.0, min(1.0, base_reward))  # Clamp to [0, 1]


def _check_reasoning_format(response: str) -> float:
    """Check if response follows proper reasoning format."""
    bonus = 0.0

    # Check for thinking tags
    if '<think>' in response and '</think>' in response:
        bonus += 0.05

    # Check for answer tags
    if '<answer>' in response and '</answer>' in response:
        bonus += 0.05

    # Check for search tags (if search was needed)
    if '<search>' in response and '</search>' in response:
        bonus += 0.05

    return bonus


def _check_search_effectiveness(response: str) -> float:
    """Check if searches were used effectively."""
    bonus = 0.0

    # Count search queries
    search_count = response.count('<search>')

    if search_count > 0:
        # Bonus for reasonable number of searches (1-3)
        if 1 <= search_count <= 3:
            bonus += 0.1
        elif search_count > 3:
            # Penalty for too many searches
            bonus -= 0.05

    # Check if information was used (presence of <information> tags)
    if '<information>' in response:
        bonus += 0.05

    return bonus


def _is_malformed_response(response: str) -> bool:
    """Check if response is malformed."""
    # Check for incomplete tags
    open_tags = ['<think>', '<search>', '<answer>', '<information>']
    close_tags = ['</think>', '</search>', '</answer>', '</information>']

    for open_tag, close_tag in zip(open_tags, close_tags):
        open_count = response.count(open_tag)
        close_count = response.count(close_tag)
        if open_count != close_count:
            return True

    return False


# For compatibility with verl's reward system
def search_r1_reward_fn(response: str, golden_answers: List[str], **kwargs) -> float:
    """Reward function that matches verl's expected interface."""
    return compute_score(response, golden_answers, **kwargs)

