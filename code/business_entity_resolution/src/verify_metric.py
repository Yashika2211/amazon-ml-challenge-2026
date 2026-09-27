"""Verify that our F0.5 metric implementation matches the competition exactly."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from metrics import f05, macro_f05
from typing import Dict, Set

print("=" * 80)
print("VERIFYING F0.5 METRIC IMPLEMENTATION")
print("=" * 80)

# Test cases with known results
test_cases = [
    {
        "name": "Perfect match",
        "truth": {"a": {"m1", "m2"}},
        "pred": {"a": {"m1", "m2"}},
        "expected_f05": 1.0
    },
    {
        "name": "No match (empty prediction)",
        "truth": {"a": set()},
        "pred": {"a": set()},
        "expected_f05": 1.0  # Empty truth, empty pred = perfect
    },
    {
        "name": "Single FP",
        "truth": {"a": {"m1"}},
        "pred": {"a": {"m1", "m2"}},
        "expected_f05": None  # Calculate manually
    },
    {
        "name": "Single FN",
        "truth": {"a": {"m1", "m2"}},
        "pred": {"a": {"m1"}},
        "expected_f05": None  # Calculate manually
    },
    {
        "name": "Multiple S1",
        "truth": {"a": {"m1", "m2"}, "b": {"m3"}},
        "pred": {"a": {"m1", "m2"}, "b": {"m3"}},
        "expected_f05": 1.0
    },
    {
        "name": "Mixed performance",
        "truth": {"a": {"m1", "m2"}, "b": {"m3"}, "c": set()},
        "pred": {"a": {"m1"}, "b": {"m3", "m4"}, "c": set()},
        "expected_f05": None  # Calculate manually
    }
]

print("\n=== TESTING F0.5 CALCULATION ===\n")

def calculate_expected_f05(tp, fp, fn):
    """Calculate F0.5 manually."""
    if tp + fp == 0 and tp + fn == 0:
        return 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0
    beta = 0.5
    f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return f05


for i, test in enumerate(test_cases, 1):
    print(f"Test {i}: {test['name']}")
    print(f"  Truth: {test['truth']}")
    print(f"  Pred:  {test['pred']}")
    
    # Calculate with our implementation
    actual_f05 = macro_f05(test["truth"], test["pred"])
    
    # Calculate expected if not provided
    if test["expected_f05"] is None:
        # Calculate per-S1 metrics
        per_s1_f05 = []
        for s1_id in test["truth"].keys():
            truth_set = test["truth"][s1_id]
            pred_set = test["pred"].get(s1_id, set())
            
            tp = len(truth_set & pred_set)
            fp = len(pred_set - truth_set)
            fn = len(truth_set - pred_set)
            
            if len(truth_set) == 0:
                # Empty truth case
                f = 1.0 if len(pred_set) == 0 else 0.0
            else:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                if precision + recall == 0:
                    f = 0.0
                else:
                    f = 1.25 * precision * recall / (0.25 * precision + recall)
            
            per_s1_f05.append(f)
            print(f"    S1 '{s1_id}': TP={tp}, FP={fp}, FN={fn}, F0.5={f:.4f}")
        
        expected_f05 = sum(per_s1_f05) / len(per_s1_f05)
    else:
        expected_f05 = test["expected_f05"]
    
    print(f"  Expected F0.5: {expected_f05:.6f}")
    print(f"  Actual F0.5:   {actual_f05:.6f}")
    
    if abs(actual_f05 - expected_f05) < 1e-6:
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL - Difference: {abs(actual_f05 - expected_f05):.8f}")
    
    print()

# Test edge cases
print("=" * 80)
print("EDGE CASE TESTING")
print("=" * 80)

edge_cases = [
    ("Empty truth, empty pred", {}, {}),
    ("Empty truth, some pred", {}, {"a": {"m1"}}),
    ("Some truth, empty pred", {"a": {"m1"}}, {}),
    ("Large TP", {"a": set(f"m{i}" for i in range(100))}, {"a": set(f"m{i}" for i in range(100))}),
    ("Large FP", {"a": {"m1"}}, {"a": set(f"m{i}" for i in range(100))}),
    ("Large FN", {"a": set(f"m{i}" for i in range(100))}, {"a": {"m1"}}),
]

for name, truth, pred in edge_cases:
    try:
        f = macro_f05(truth, pred)
        print(f"{name:30s} F0.5={f:.6f}")
    except Exception as e:
        print(f"{name:30s} ERROR: {e}")

print("\n" + "=" * 80)
print("FORMULA VERIFICATION")
print("=" * 80)

# Verify the formula matches: F0.5 = 1.25 * P * R / (0.25 * P + R)
print("\nF0.5 Formula: (1 + β²) * P * R / (β² * P + R) where β = 0.5")
print("Simplifies to: 1.25 * P * R / (0.25 * P + R)")
print()

beta = 0.5
beta_sq = beta ** 2
coef = 1 + beta_sq
denom_coef = beta_sq

print(f"β = {beta}")
print(f"β² = {beta_sq}")
print(f"(1 + β²) = {coef}")
print(f"Denominator coefficient = β² = {denom_coef}")
print()

# Test with specific precision/recall values
test_values = [
    (1.0, 1.0),  # Perfect
    (1.0, 0.5),  # High precision, lower recall
    (0.5, 1.0),  # Lower precision, high recall
    (0.9, 0.8),  # High both
    (0.5, 0.5),  # Medium both
]

print("Sample F0.5 values:")
print(f"{'Precision':<12} {'Recall':<12} {'F0.5':<12}")
print("-" * 40)

for p, r in test_values:
    if p + r > 0:
        f05_val = coef * p * r / (denom_coef * p + r)
        print(f"{p:<12.2f} {r:<12.2f} {f05_val:<12.6f}")

print("\n" + "=" * 80)
print("METRIC VERIFICATION COMPLETE")
print("=" * 80)
print("\nConclusion: Verify that all test cases pass and formula matches competition specs.")
print("If any failures, update src/metrics.py accordingly.")
