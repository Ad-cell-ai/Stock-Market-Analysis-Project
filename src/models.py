from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_model(model_name="rf"):
    """Return ML model by name."""
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    elif model_name == "xgb":
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
    else:
        raise ValueError("Choose model: 'rf' or 'xgb'")
