"""Arithmetic reward with a small length term to exercise nonzero GRPO updates.

The length term is integration-test shaping, not a quality benchmark score.
"""

import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    answers = re.findall(r"-?\d+", solution_str)
    correct = float(bool(answers) and answers[-1] == ground_truth)
    return correct + 0.01 / (1 + len(solution_str))
