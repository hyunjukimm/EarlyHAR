"""
TEASER/MUSE wrapper using sktime for early time series classification.
Falls back to simple 1-NN if sktime not available.
"""
import numpy as np

try:
    from sktime.classification.early_classification import TEASER
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    from sktime.datatypes import convert_to
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False


def _to_nested_univ(X):
    """Convert (N, T, C) to sktime nested_univ format."""
    from sktime.datatypes import convert_to
    return convert_to(X, to_type="nested_univ")


def create_teaser_model():
    if HAS_SKTIME:
        return TEASER(
            random_state=42,
            classification_points=[0.25, 0.5, 0.75, 1.0],
            estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=42),
        )
    return None


def fit_teaser(model, X_train, y_train):
    """X_train: (N, T, C)."""
    if model is None:
        return None
    try:
        X_conv = _to_nested_univ(X_train)
        model.fit(X_conv, y_train)
        return model
    except Exception as e:
        print(f"TEASER fit failed: {e}")
        return None


def predict_teaser(model, X_test):
    """Predict and return (y_pred, earliness). Earliness = 1 - (decisions/T)."""
    if model is None:
        return None, None
    try:
        X_conv = _to_nested_univ(X_test)
        y_pred, decisions = model.predict(X_conv)
        T = X_test.shape[1]
        earliness = 1.0 - (np.mean(decisions) / T) if T > 0 else 1.0
        return y_pred, earliness
    except Exception as e:
        print(f"TEASER predict failed: {e}")
        return None, None


def fallback_1nn_predict(X_train, y_train, X_test):
    """Simple 1-NN on flattened series as fallback."""
    from sklearn.neighbors import KNeighborsClassifier
    N_tr, T, C = X_train.shape
    N_te = X_test.shape[0]
    X_tr_flat = X_train.reshape(N_tr, -1)
    X_te_flat = X_test.reshape(N_te, -1)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_tr_flat, y_train)
    y_pred = clf.predict(X_te_flat)
    earliness = 1.0
    return y_pred, earliness
