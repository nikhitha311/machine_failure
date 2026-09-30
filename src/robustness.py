def robustness_report(baseline_metrics, robust_metrics):
    """Compute shift drop and improvement."""
    base_f1 = baseline_metrics["f1"]
    rob_f1 = robust_metrics["f1"]
    improvement = ((rob_f1 - base_f1) / base_f1 * 100.0) if base_f1 > 0 else 0.0
    return {
        "baseline_f1": base_f1,
        "robust_f1": rob_f1,
        "improvement_pct": improvement,
    }