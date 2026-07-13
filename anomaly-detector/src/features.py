"""
Transforma o historico bruto do Zabbix (lista de {clock, value}) em um
DataFrame com features prontas para o modelo de deteccao de anomalias.
"""
import pandas as pd


def history_to_dataframe(history: list) -> pd.DataFrame:
    """Converte a resposta de history.get em um DataFrame ordenado por tempo."""
    if not history:
        return pd.DataFrame(columns=["timestamp", "value"])

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["clock"].astype(int), unit="s")
    df["value"] = df["value"].astype(float)
    df = df[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)
    return df


def resample_series(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Reamostra a serie para um intervalo fixo, preenchendo buracos por interpolacao."""
    if df.empty:
        return df
    s = df.set_index("timestamp")["value"].resample(freq).mean()
    s = s.interpolate(limit_direction="both")
    return s.reset_index()


def build_features(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Recebe um DataFrame com colunas [timestamp, value] (ja reamostrado) e
    adiciona features usadas pelo modelo:
      - rolling_mean / rolling_std: padrao recente da serie
      - deviation: quanto o valor atual foge da media movel
      - hour / dayofweek: contexto temporal (padroes variam por horario/dia)
    """
    out = df.copy()
    out["rolling_mean"] = out["value"].rolling(window, min_periods=1).mean()
    out["rolling_std"] = out["value"].rolling(window, min_periods=1).std().fillna(0)
    out["deviation"] = out["value"] - out["rolling_mean"]
    out["hour"] = out["timestamp"].dt.hour
    out["dayofweek"] = out["timestamp"].dt.dayofweek
    return out


FEATURE_COLUMNS = ["value", "rolling_mean", "rolling_std", "deviation", "hour", "dayofweek"]


def feature_matrix(df_features: pd.DataFrame):
    return df_features[FEATURE_COLUMNS].fillna(0)
