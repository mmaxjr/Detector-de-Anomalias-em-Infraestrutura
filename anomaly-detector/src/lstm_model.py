"""
Deteccao de anomalias com autoencoder LSTM (rede neural recorrente).

Ideia: a rede aprende a "reconstruir" pequenas janelas (sequencias) da
serie normal. Quando o erro de reconstrucao de uma janela nova e muito
maior que o normal, isso indica um padrao que a rede nunca viu -
inclusive padroes que dependem do MOMENTO (ex: um pico de CPU que e
normal as 8h da manha mas anomalo as 3h da madrugada), coisa que o
Isolation Forest simples enxerga com menos nuance.

Mais pesado para treinar (precisa de tensorflow) e um pouco mais lento
que o Isolation Forest, mas mais sensivel a padroes temporais sutis.
"""
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from src.features import build_features, FEATURE_COLUMNS

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(X) - seq_len + 1
    if n <= 0:
        return np.empty((0, seq_len, X.shape[1]))
    return np.stack([X[i:i + seq_len] for i in range(n)])


def _build_autoencoder(seq_len: int, n_features: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(seq_len, n_features))
    encoded = layers.LSTM(16, activation="tanh")(inputs)
    repeated = layers.RepeatVector(seq_len)(encoded)
    decoded = layers.LSTM(16, activation="tanh", return_sequences=True)(repeated)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(decoded)

    autoencoder = keras.Model(inputs, outputs)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train(df_resampled, window: int = 12, seq_len: int = 12, epochs: int = 20, percentile: float = 98.0):
    """
    Treina o autoencoder sobre a serie reamostrada e ja resampleada.
    Retorna (keras_model, scaler, threshold, df_com_score_e_anomalia).
    """
    df_feat = build_features(df_resampled, window=window)
    X_raw = df_feat[FEATURE_COLUMNS].fillna(0).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    seqs = _sequences(X, seq_len)
    if len(seqs) < 20:
        raise ValueError(
            "Poucos dados para treinar o LSTM (minimo ~20 sequencias). "
            "Use um periodo historico maior na aba Treino."
        )

    model = _build_autoencoder(seq_len, X.shape[1])
    model.fit(seqs, seqs, epochs=epochs, batch_size=32, verbose=0, shuffle=True)

    recon = model.predict(seqs, verbose=0)
    errors = np.mean((seqs - recon) ** 2, axis=(1, 2))
    threshold = float(np.percentile(errors, percentile))

    # cada sequencia representa uma janela; associamos o score ao timestamp
    # do ULTIMO ponto da janela (os primeiros seq_len-1 pontos ficam sem score)
    df_result = df_feat.iloc[seq_len - 1:].reset_index(drop=True).copy()
    df_result["anomaly_score"] = errors
    df_result["is_anomaly"] = errors > threshold

    return model, scaler, threshold, df_result


def detect(df_resampled, keras_model, scaler, threshold, window: int = 12, seq_len: int = 12):
    df_feat = build_features(df_resampled, window=window)
    X_raw = df_feat[FEATURE_COLUMNS].fillna(0).values
    X = scaler.transform(X_raw)

    seqs = _sequences(X, seq_len)
    if len(seqs) == 0:
        df_feat["anomaly_score"] = 0.0
        df_feat["is_anomaly"] = False
        return df_feat

    recon = keras_model.predict(seqs, verbose=0)
    errors = np.mean((seqs - recon) ** 2, axis=(1, 2))

    df_result = df_feat.iloc[seq_len - 1:].reset_index(drop=True).copy()
    df_result["anomaly_score"] = errors
    df_result["is_anomaly"] = errors > threshold
    return df_result


def _safe_name(model_name: str) -> str:
    """Mesma sanitizacao usada em src/model.py, para o nome do arquivo bater
    nos dois lugares (senao o load_model nao encontra o arquivo salvo aqui)."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)


def save(keras_model, scaler, threshold, seq_len, window, model_name, meta=None, epochs=None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_name = _safe_name(model_name)
    lstm_dir = os.path.join(MODELS_DIR, f"{safe_name}_lstm")
    os.makedirs(lstm_dir, exist_ok=True)
    keras_model.save(os.path.join(lstm_dir, "model.keras"))

    manifest = {
        "type": "lstm",
        "lstm_dir": lstm_dir,
        "scaler": scaler,
        "threshold": threshold,
        "seq_len": seq_len,
        "window": window,
        "epochs": epochs,
        "meta": meta or {},
    }
    path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
    joblib.dump(manifest, path)
    return path


def load_keras_model(manifest: dict):
    from tensorflow import keras
    return keras.models.load_model(os.path.join(manifest["lstm_dir"], "model.keras"))
