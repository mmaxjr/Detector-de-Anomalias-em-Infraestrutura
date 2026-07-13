"""
Combinacao de multiplas metricas (do mesmo host) em um unico DataFrame
alinhado no tempo, para deteccao de anomalias multivariada - ou seja,
que considera a CORRELACAO entre metricas (ex: CPU alta + rede alta +
disco cheio ao mesmo tempo) em vez de olhar cada uma isoladamente.
"""
import pandas as pd

from src.features import history_to_dataframe, resample_series


def combine_histories(histories: dict, freq: str = "5min") -> pd.DataFrame:
    """
    histories: {label: historico_bruto_do_zabbix}
    Retorna um DataFrame [timestamp, <label1>, <label2>, ...] com todas as
    series reamostradas no mesmo intervalo e alinhadas pelo tempo.
    """
    series = {}
    for label, hist in histories.items():
        df_raw = history_to_dataframe(hist)
        df_res = resample_series(df_raw, freq=freq)
        if df_res.empty:
            continue
        series[label] = df_res.set_index("timestamp")["value"]

    if not series:
        return pd.DataFrame()

    combined = pd.DataFrame(series)
    combined = combined.interpolate(limit_direction="both").dropna()
    combined = combined.reset_index().rename(columns={"index": "timestamp"})
    return combined


def build_multi_features(df_combined: pd.DataFrame, value_columns: list, window: int = 12) -> pd.DataFrame:
    """Adiciona media movel e desvio para cada metrica, mais contexto temporal."""
    out = df_combined.copy()
    for col in value_columns:
        out[f"{col}__roll_mean"] = out[col].rolling(window, min_periods=1).mean()
        out[f"{col}__dev"] = out[col] - out[f"{col}__roll_mean"]
    out["hour"] = out["timestamp"].dt.hour
    out["dayofweek"] = out["timestamp"].dt.dayofweek
    return out


def multi_feature_columns(value_columns: list) -> list:
    cols = []
    for col in value_columns:
        cols += [col, f"{col}__roll_mean", f"{col}__dev"]
    cols += ["hour", "dayofweek"]
    return cols
