"""
CaviAI - XGBoost Cavity Risk Model
Predicts personalized cavity risk score and recommended checkup timing.
"""


def save_model(model, path: str = "caviai_model.joblib"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")
 
def load_model(path: str = "caviai_model.joblib"):
    return joblib.load(path)


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
 
    # Synthetic dataset — replace with real patient records
    raw = pd.DataFrame({
        "record_date":              pd.date_range("2020-01-01", periods=n, freq="3D"),
        "cavity_count":             np.random.poisson(2, n),
        "years_since_last":         np.random.uniform(0, 5, n),
        "sugary_drinks_per_day":    np.random.uniform(0, 5, n),
        "sweets_per_week":          np.random.uniform(0, 14, n),
        "acid_drinks_per_day":      np.random.uniform(0, 4, n),
        "brushing_per_day":         np.random.choice([1, 2, 3], n),
        "flosses_per_week":         np.random.choice([0, 1, 3, 7], n),
        "fluoride_toothpaste":      np.random.randint(0, 2, n),
        "mouthwash_per_week":       np.random.choice([0, 2, 7], n),
        "family_cavity_history":    np.random.choice([0, 1, 2], n),
        "dry_mouth":                np.random.randint(0, 2, n),
        "age":                      np.random.randint(18, 75, n),
        "months_since_last_checkup": np.random.uniform(0, 36, n),
        "cavity_within_12mo":       np.random.randint(0, 2, n),   # label
    })
 
    print("Training CaviAI XGBoost model...")
    model = train(raw, target_col="cavity_within_12mo")
 
    save_model(model)
 
    # Score a sample user
    sample_user = {
        "cavity_count": 4,
        "years_since_last": 0.5,         # recent cavity
        "sugary_drinks_per_day": 3,
        "sweets_per_week": 10,
        "acid_drinks_per_day": 2,
        "brushing_per_day": 1,           # only brushes once a day
        "flosses_per_week": 0,           # never flosses
        "fluoride_toothpaste": 0,
        "mouthwash_per_week": 0,
        "family_cavity_history": 2,      # strong family history
        "dry_mouth": 1,
        "age": 34,
        "months_since_last_checkup": 18, # overdue
    }
 
    result = score_user(model, sample_user)
 
    print("\n── CaviAI Risk Profile ──────────────────")
    print(f"Risk Score     : {result['risk_score']}%")
    print(f"Risk Band      : {result['risk_band']}")
    print(f"Next Checkup   : ~{result['next_checkup_months']} months")
    print(f"Message        : {result['message']}")
    print("\nTop Risk Drivers:")
    for d in result["top_drivers"]:
        print(f"  {d['feature']:<28} {d['contribution']}  (SHAP: {d['shap_value']:+.3f})")
