def train(df: pd.DataFrame, target_col: str = "cavity_within_12mo"):
    """
    Time-series cross-validation: train on past, validate on future.
    Critical for health models — standard k-fold leaks future information.
 
    Args:
        df          : must include a 'record_date' column for ordering
        target_col  : binary label — did the user develop a cavity within 12 months?
    """
    df = df.sort_values("record_date").reset_index(drop=True)
    X = engineer_features(df)[FEATURE_COLS]
    y = df[target_col]
 
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores, brier_scores = [], []
 
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
 
        model = build_model()
        model.fit(X_train, y_train)
 
        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        brier = brier_score_loss(y_val, probs)
 
        auc_scores.append(auc)
        brier_scores.append(brier)
        print(f"Fold {fold+1}: AUC={auc:.3f}  Brier={brier:.3f}")
 
    print(f"\nMean AUC:   {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    print(f"Mean Brier: {np.mean(brier_scores):.3f} ± {np.std(brier_scores):.3f}")
 
    # Final model trained on all data
    final_model = build_model()
    final_model.fit(X, y)
    return final_model
 
 
# ─────────────────────────────────────────────
# 4. RISK SCORING & CHECKUP TIMING
# ─────────────────────────────────────────────
 
RISK_BANDS = [
    (0.00, 0.25, "Low",      18, "green"),
    (0.25, 0.55, "Moderate", 9,  "yellow"),
    (0.55, 0.80, "High",     5,  "orange"),
    (0.80, 1.00, "Critical", 2,  "red"),
]
 
def score_user(model, user_data: dict) -> dict:
    """
    Score a single user and return their risk profile.
 
    Args:
        model     : trained CalibratedClassifierCV
        user_data : dict matching the raw input schema
 
    Returns:
        dict with risk_score, risk_band, next_checkup_months, top_drivers
    """
    df_user = pd.DataFrame([user_data])
    X_user = engineer_features(df_user)[FEATURE_COLS]
 
    risk_score = float(model.predict_proba(X_user)[0, 1])
 
    # Determine band and recommended checkup interval
    band_label, checkup_months, band_color = "Unknown", 6, "gray"
    for lo, hi, label, months, color in RISK_BANDS:
        if lo <= risk_score < hi:
            band_label, checkup_months, band_color = label, months, color
            break
 
    # SHAP values for explainability — tell users *why* they're high risk
    explainer = shap.TreeExplainer(
        model.calibrated_classifiers_[0].estimator.named_steps["xgb"]
    )
    # Pass through scaler before SHAP
    X_scaled = model.calibrated_classifiers_[0].estimator.named_steps["scaler"].transform(X_user)
    shap_vals = explainer.shap_values(X_scaled)[0]
 
    # Top 3 risk drivers sorted by absolute SHAP value
    driver_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "shap_value": shap_vals,
        "contribution": ["↑ Risk" if v > 0 else "↓ Risk" for v in shap_vals],
    }).reindex(pd.Index(np.argsort(np.abs(shap_vals))[::-1]))
 
    top_drivers = driver_df.head(3)[["feature", "shap_value", "contribution"]].to_dict("records")
 
    return {
        "risk_score": round(risk_score * 100, 1),   # e.g. 68.4
        "risk_band": band_label,
        "band_color": band_color,
        "next_checkup_months": checkup_months,
        "top_drivers": top_drivers,
        "message": _build_message(band_label, checkup_months, top_drivers),
    }
 
 
def _build_message(band: str, months: int, drivers: list) -> str:
    top = drivers[0]["feature"].replace("_", " ") if drivers else "your habits"
    return (
        f"Your cavity risk is {band.lower()}. "
        f"We recommend your next checkup in ~{months} months. "
        f"Your biggest risk factor is {top}."
    )
 
 
# ─────────────────────────────────────────────
# 5. CALIBRATION PLOT
# ─────────────────────────────────────────────
 
def plot_calibration(model, X_val, y_val, save_path: str = "calibration.png"):
    """
    A well-calibrated model is non-negotiable for a health product.
    If the curve drifts far from the diagonal, retrain or recalibrate.
    """
    probs = model.predict_proba(X_val)[:, 1]
    fraction_pos, mean_predicted = calibration_curve(y_val, probs, n_bins=10)
 
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_predicted, fraction_pos, "o-", label="CaviAI model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Calibration plot saved to {save_path}")
