"""
Verificacao unificada de um modelo salvo (qualquer tipo: single, multi ou
lstm) contra dados recentes do Zabbix. Usado tanto pela aba Monitoramento
(um modelo por vez, com grafico) quanto pela aba Dashboard (varios modelos
de uma vez, em forma de mapa de calor).
"""
from datetime import datetime, timezone

from src.features import history_to_dataframe, resample_series
from src.multi_features import combine_histories
from src import model as model_lib
from src import lstm_model


def _no_data_message(client, item_id: str) -> str:
    """
    Quando history.get nao retorna nada, consulta item.get para descobrir
    quando foi a ULTIMA coleta real desse item. Se essa data for recente
    (mas ainda assim fora da janela pedida), o provavel motivo e relogio
    dessincronizado entre esta maquina e o servidor Zabbix - nao falta de
    coleta.
    """
    base_msg = "Sem dados no periodo."
    try:
        status = client.get_item_status(item_id)
    except AttributeError:
        return (base_msg + " (conexao antiga em memoria - va na aba 'Conexao' e clique em "
                "'Conectar e salvar' de novo para atualizar)")
    except Exception as e:
        return f"{base_msg} (nao foi possivel checar ultima coleta: {e})"

    lastclock_raw = status.get("lastclock") if status else None
    if not lastclock_raw or int(lastclock_raw) == 0:
        return (base_msg + " O Zabbix reporta que este item nunca teve historico registrado "
                "(lastclock=0) - confira na tela 'Ultimos dados' do Zabbix se esse item realmente "
                "esta coletando.")

    last_dt = datetime.fromtimestamp(int(lastclock_raw), tz=timezone.utc)
    now_dt = datetime.now(timezone.utc)
    diff_min = (now_dt - last_dt).total_seconds() / 60

    last_str = last_dt.strftime("%Y-%m-%d %H:%M UTC")
    if diff_min < 0:
        return f"{base_msg} Ultima coleta real: {last_str} (no FUTURO em relacao ao relogio local - verifique o horario da maquina)."
    elif diff_min > 24 * 60:
        return f"{base_msg} Ultima coleta real: {last_str} (ha mais de 1 dia - item pode ter parado de coletar)."
    else:
        return f"{base_msg} Ultima coleta real: {last_str} (recente!) - se a janela pedida nao pegou isso, verifique o relogio da maquina que roda o Streamlit."


def check_model(client, model_name: str, hours: int = 6):
    """
    Retorna um dict:
      {
        "status": "green" | "yellow" | "red" | "gray",
        "message": str,
        "n_anomalies": int,
        "df": DataFrame ou None,
        "value_columns": list[str] (para modelos multi-metrica),
        "type": str,
      }
    """
    payload = model_lib.load_model(model_name)
    if payload is None:
        return {"status": "gray", "message": "Modelo nao encontrado.", "n_anomalies": 0, "df": None,
                "value_columns": [], "type": "single"}

    meta = payload.get("meta", {})
    model_type = payload.get("type", "single")
    window = payload.get("window", 12)
    time_from = int(client.days_ago_timestamp(0) - hours * 3600)

    try:
        if model_type == "multi":
            items = meta.get("items", [])
            if not items:
                return {"status": "gray", "message": "Modelo multi-metrica sem itens configurados.",
                        "n_anomalies": 0, "df": None, "value_columns": [], "type": model_type}

            histories = {it["key"]: client.get_history(it["item_id"], it["value_type"], time_from) for it in items}
            df_combined = combine_histories(histories)
            if df_combined.empty:
                msg = _no_data_message(client, items[0]["item_id"])
                return {"status": "gray", "message": msg, "n_anomalies": 0, "df": None,
                        "value_columns": [], "type": model_type}

            value_columns = [it["key"] for it in items]
            df_result = model_lib.detect_multi(df_combined, payload["model"], value_columns, window=window)

        elif model_type == "lstm":
            if not meta.get("item_id"):
                return {"status": "gray", "message": "Modelo LSTM sem item configurado.", "n_anomalies": 0,
                        "df": None, "value_columns": [], "type": model_type}
            history = client.get_history(meta["item_id"], meta["value_type"], time_from)
            if not history:
                msg = _no_data_message(client, meta["item_id"])
                return {"status": "gray", "message": msg, "n_anomalies": 0, "df": None,
                        "value_columns": ["value"], "type": model_type}
            df_raw = history_to_dataframe(history)
            df_resampled = resample_series(df_raw, freq="5min")
            df_result = lstm_model.detect(
                df_resampled, payload["keras_model"], payload["scaler"], payload["threshold"],
                window=window, seq_len=payload["seq_len"],
            )
            value_columns = ["value"]

        else:  # "single"
            if not meta.get("item_id"):
                return {"status": "gray", "message": "Modelo antigo sem item_id. Retreine e salve novamente.",
                        "n_anomalies": 0, "df": None, "value_columns": ["value"], "type": model_type}
            history = client.get_history(meta["item_id"], meta["value_type"], time_from)
            if not history:
                msg = _no_data_message(client, meta["item_id"])
                return {"status": "gray", "message": msg, "n_anomalies": 0, "df": None,
                        "value_columns": ["value"], "type": model_type}
            df_raw = history_to_dataframe(history)
            df_resampled = resample_series(df_raw, freq="5min")
            df_result = model_lib.detect(df_resampled, payload["model"], window=window)
            value_columns = ["value"]

        if df_result is None or df_result.empty:
            return {"status": "gray", "message": "Sem pontos suficientes para avaliar.", "n_anomalies": 0,
                    "df": None, "value_columns": value_columns, "type": model_type}

        n_anom = int(df_result["is_anomaly"].sum())
        last_anom = bool(df_result.iloc[-1]["is_anomaly"])

        if last_anom:
            status = "red"
            message = f"Ultimo ponto fora do padrao ({n_anom} anomalias em {hours}h)"
        elif n_anom > 0:
            status = "yellow"
            message = f"{n_anom} anomalias em {hours}h, ultimo ponto normal"
        else:
            status = "green"
            message = f"Normal nas ultimas {hours}h"

        return {"status": status, "message": message, "n_anomalies": n_anom, "df": df_result,
                "value_columns": value_columns, "type": model_type}

    except Exception as e:
        return {"status": "gray", "message": f"Erro: {e}", "n_anomalies": 0, "df": None,
                "value_columns": [], "type": model_type}
