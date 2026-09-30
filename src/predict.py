
import numpy as np


def predict_one(model, sample_df, feature_names,
                sensor_cols, top_k=5):
    """Predict failure probability for one sample."""
    X = sample_df.reindex(columns=feature_names, fill_value=0)
    prob = float(model.predict_proba(X.values)[0, 1])
    pred = int(prob >= 0.5)

    if prob < 0.30:
        risk = "NORMAL"
    elif prob < 0.70:
        risk = "WARNING"
    else:
        risk = "CRITICAL"

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_k]
    top_features = [
        (feature_names[i], float(importances[i])) for i in order
    ]

    row = X.values[0]
    contributions = []
    for i in order:
        col = feature_names[i]
        val = row[i]
        contributions.append((col, float(importances[i]), float(val)))

    explanation = _explain(contributions, sensor_cols)
    recommendation = _recommend(prob, contributions)

    return {
        "probability": prob,
        "prediction": pred,
        "risk_level": risk,
        "top_features": top_features,
        "explanation": explanation,
        "recommendation": recommendation,
    }


def _explain(contributions, sensor_cols):
    """Rule-based text explanation."""
    reverse = {v: k for k, v in sensor_cols.items()}
    lines = []
    for col, imp, val in contributions[:4]:
        canon = reverse.get(col, col)
        if canon == "torque":
            lines.append(f"Torque influence (value={val:.2f}, importance={imp:.3f})")
        elif canon in ("air_temp", "process_temp"):
            lines.append(f"Temperature influence (value={val:.2f}, importance={imp:.3f})")
        elif canon == "tool_wear":
            lines.append(f"Tool wear influence (value={val:.2f}, importance={imp:.3f})")
        elif canon == "rpm":
            lines.append(f"RPM deviation influence (value={val:.2f}, importance={imp:.3f})")
        else:
            lines.append(f"{col} influence (value={val:.2f}, importance={imp:.3f})")
    return lines


def _recommend(prob, contributions):
    if prob >= 0.70:
        return ("CRITICAL: Stop/inspect machine immediately. "
                "Check bearings, cooling system, and tool wear.")
    elif prob >= 0.30:
        return ("WARNING: Schedule inspection soon. "
                "Monitor temperature, torque, and tool wear trends.")
    else:
        return "NORMAL: Continue operation. Routine monitoring only."
