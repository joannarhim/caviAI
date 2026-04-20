
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw user inputs into model-ready features.
 
    Expected raw columns:
        cavity_count        : total cavities ever
        years_since_last    : years since most recent cavity
        sugary_drinks_per_day
        sweets_per_week
        acid_drinks_per_day : (coffee, soda, juice)
        brushing_per_day
        flosses_per_week
        fluoride_toothpaste : bool (1/0)
        mouthwash_per_week
        family_cavity_history : 0=none, 1=some, 2=strong
        dry_mouth           : bool (1/0)  — saliva protects enamel
        age
        months_since_last_checkup
    """
    feat = df.copy()
 
    # --- Cavity history ---
    # Decay weighting: recent cavities matter more than old ones
    feat["cavity_rate"] = feat["cavity_count"] / feat["age"].clip(lower=1)
    feat["recency_weight"] = np.exp(-0.3 * feat["years_since_last"].clip(lower=0))
    feat["weighted_cavity_score"] = feat["cavity_count"] * feat["recency_weight"]
 
    # --- Diet: fermentable carb & acid exposure ---
    feat["sugar_exposure"] = (
        feat["sugary_drinks_per_day"] * 1.5   # liquid sugar coats teeth
        + feat["sweets_per_week"] / 7
    )
    feat["acid_exposure"] = feat["acid_drinks_per_day"]
    feat["diet_risk_score"] = feat["sugar_exposure"] + feat["acid_exposure"] * 0.8
 
    # --- Hygiene composite (higher = better) ---
    feat["hygiene_score"] = (
        feat["brushing_per_day"] * 2.0
        + feat["flosses_per_week"] / 7 * 1.5
        + feat["fluoride_toothpaste"] * 2.0
        + feat["mouthwash_per_week"] / 7 * 0.5
    )
    feat["hygiene_score"] = feat["hygiene_score"].clip(upper=10)  # normalize
 
    # Net risk = diet risk penalized by hygiene
    feat["net_oral_risk"] = feat["diet_risk_score"] / (feat["hygiene_score"].clip(lower=1))
 
    # --- Genetic / biological risk proxy ---
    # family_cavity_history encodes genetic susceptibility
    feat["genetic_risk"] = feat["family_cavity_history"] + feat["dry_mouth"] * 1.5
 
    # --- Overdue factor ---
    # ADA recommends max 12-month intervals; penalize beyond that
    feat["overdue_months"] = (feat["months_since_last_checkup"] - 12).clip(lower=0)
 
    # Drop raw columns the model doesn't need directly
    cols_to_drop = [
        "sugary_drinks_per_day", "sweets_per_week", "acid_drinks_per_day",
        "brushing_per_day", "flosses_per_week", "mouthwash_per_week",
    ]
    feat.drop(columns=cols_to_drop, inplace=True, errors="ignore")
 
    return feat

FEATURE_COLS = [
    "cavity_rate",
    "recency_weight",
    "weighted_cavity_score",
    "sugar_exposure",
    "acid_exposure",
    "diet_risk_score",
    "hygiene_score",
    "net_oral_risk",
    "genetic_risk",
    "fluoride_toothpaste",
    "dry_mouth",
    "age",
    "months_since_last_checkup",
    "overdue_months",
]


def build_model() -> Pipeline:
    """
    XGBoost wrapped in a sklearn Pipeline.
    StandardScaler is lightweight but helps when features span
    different magnitudes (e.g. age vs. daily drink count).
    """
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,           # shallow = less overfitting on small datasets
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,    # prevents fitting on tiny subgroups
        gamma=0.1,
        scale_pos_weight=3,    # adjust if cavity cases are ~25% of dataset
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb),
    ])
 
    # Isotonic calibration makes probability outputs statistically reliable
    # (i.e. "72% risk" actually means 72%, not just "higher than 60%")
    calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv=5)
    return calibrated
