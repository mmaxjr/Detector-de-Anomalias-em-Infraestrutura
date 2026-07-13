"""
Treino, persistencia e inferencia dos modelos de deteccao de anomalias.

Suporta tres tipos (campo "type" salvo no arquivo .joblib):
  - "single": Isolation Forest sobre 1 metrica (caso simples original).
  - "multi":  Isolation Forest sobre varias metricas do mesmo host ao
              mesmo tempo, capturando CORRELACAO entre elas
              (ex: CPU + rede + disco subindo juntos).
  - "lstm":   Autoencoder LSTM sobre 1 metrica (ver src/lstm_model.py).
"""
import os
import shutil
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features import build_features, feature_matrix
from src.multi_features import build_multi_features, multi_feature_columns

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def safe_name(model_name: str) -> str:
    """Sanitiza o nome do modelo para virar um nome de arquivo valido.
    Usado tanto aqui quanto em src/lstm_model.py - os dois precisam bater."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)


def _model_path(model_name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f"{safe_name(model_name)}.joblib")


# --------------------------------------------------------------------
# Isolation Forest - metrica unica
# --------------------------------------------------------------------
def train_model(df_resampled: pd.DataFrame, contamination: float = 0.02, window: int = 12):
    df_feat = build_features(df_resampled, window=window)
    X = feature_matrix(df_feat)

    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    clf.fit(X)

    df_feat["anomaly_score"] = clf.decision_function(X)
    df_feat["is_anomaly"] = clf.predict(X) == -1
    return clf, df_feat


def detect(df_resampled: pd.DataFrame, clf, window: int = 12) -> pd.DataFrame:
    df_feat = build_features(df_resampled, window=window)
    X = feature_matrix(df_feat)
    df_feat["anomaly_score"] = clf.decision_function(X)
    df_feat["is_anomaly"] = clf.predict(X) == -1
    return df_feat


# --------------------------------------------------------------------
# Isolation Forest - multi-metrica (correlacao)
# --------------------------------------------------------------------
def train_multi_model(df_combined: pd.DataFrame, value_columns: list, contamination: float = 0.02, window: int = 12):
    df_feat = build_multi_features(df_combined, value_columns, window=window)
    cols = multi_feature_columns(value_columns)
    X = df_feat[cols].fillna(0)

    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    clf.fit(X)

    df_feat["anomaly_score"] = clf.decision_function(X)
    df_feat["is_anomaly"] = clf.predict(X) == -1
    return clf, df_feat


def detect_multi(df_combined: pd.DataFrame, clf, value_columns: list, window: int = 12) -> pd.DataFrame:
    df_feat = build_multi_features(df_combined, value_columns, window=window)
    cols = multi_feature_columns(value_columns)
    X = df_feat[cols].fillna(0)
    df_feat["anomaly_score"] = clf.decision_function(X)
    df_feat["is_anomaly"] = clf.predict(X) == -1
    return df_feat


# --------------------------------------------------------------------
# Persistencia generica
# --------------------------------------------------------------------
def save_model(clf, model_name: str, window: int = 12, meta: dict = None, model_type: str = "single"):
    """Salva modelos do tipo Isolation Forest (single ou multi)."""
    path = _model_path(model_name)
    joblib.dump(
        {"type": model_type, "model": clf, "window": window, "meta": meta or {}},
        path,
    )
    return path


def load_model(model_name: str):
    path = _model_path(model_name)
    if not os.path.exists(path):
        return None
    payload = joblib.load(path)
    payload.setdefault("type", "single")

    if payload["type"] == "lstm":
        from src import lstm_model
        payload["keras_model"] = lstm_model.load_keras_model(payload)

    return payload


def list_saved_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    return sorted(f[: -len(".joblib")] for f in os.listdir(MODELS_DIR) if f.endswith(".joblib"))


def delete_model(model_name: str) -> bool:
    """Remove um modelo salvo (e a pasta do LSTM, se houver)."""
    path = _model_path(model_name)
    if not os.path.exists(path):
        return False

    try:
        payload = joblib.load(path)
        if payload.get("type") == "lstm" and payload.get("lstm_dir") and os.path.isdir(payload["lstm_dir"]):
            shutil.rmtree(payload["lstm_dir"], ignore_errors=True)
    except Exception:
        pass

    os.remove(path)
    return True


def rename_model(old_name: str, new_name: str):
    """Renomeia um modelo salvo. Retorna (sucesso: bool, mensagem: str)."""
    old_path = _model_path(old_name)
    new_path = _model_path(new_name)

    if not os.path.exists(old_path):
        return False, "Modelo original nao encontrado."
    if os.path.exists(new_path):
        return False, "Ja existe um modelo salvo com esse nome."

    payload = joblib.load(old_path)

    if payload.get("type") == "lstm" and payload.get("lstm_dir"):
        old_dir = payload["lstm_dir"]
        new_dir = os.path.join(MODELS_DIR, f"{safe_name(new_name)}_lstm")
        if os.path.isdir(old_dir):
            shutil.move(old_dir, new_dir)
        payload["lstm_dir"] = new_dir

    joblib.dump(payload, new_path)
    os.remove(old_path)
    return True, "Modelo renomeado com sucesso."
