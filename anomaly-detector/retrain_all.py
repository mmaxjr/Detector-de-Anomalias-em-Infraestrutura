"""
Retreino automatico de todos os modelos salvos.
--------------------------------------------------
Script standalone (nao depende da interface Streamlit aberta) que:
  1. Le a conexao salva com o Zabbix (a mesma usada pelo app.py).
  2. Para cada modelo salvo, busca dados novos e retreina do zero,
     reusando os mesmos parametros (dias de historico, sensibilidade,
     epocas do LSTM) que foram usados no treino original.
  3. Sobrescreve o modelo salvo com a versao retreinada.

Uso manual:
    python retrain_all.py

Uso agendado (recomendado - ver README.md para o passo a passo com o
Agendador de Tarefas do Windows):
    Rodar esse script 1x por semana mantem os modelos alinhados com o
    padrao normal mais recente da infraestrutura, sem precisar abrir
    a interface grafica.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.zabbix_client import ZabbixClient
from src.credentials import load_connection
from src.features import history_to_dataframe, resample_series
from src.multi_features import combine_histories
from src import model as model_lib
from src import lstm_model


def get_client() -> ZabbixClient:
    conn = load_connection()
    if not conn:
        raise RuntimeError(
            "Nenhuma conexao salva encontrada. Abra o app (streamlit run app.py) e "
            "conecte-se ao Zabbix pelo menos uma vez antes de usar este script."
        )
    if conn.get("auth_mode") == "token":
        return ZabbixClient(url=conn["url"], token=conn.get("token"))
    return ZabbixClient(url=conn["url"], user=conn.get("user"), password=conn.get("password"))


def _retrain_single(client: ZabbixClient, name: str, payload: dict) -> bool:
    meta = payload.get("meta", {})
    item_id = meta.get("item_id")
    value_type = meta.get("value_type")
    if not item_id:
        print(f"  [pular] sem item_id salvo neste modelo.")
        return False

    dias = meta.get("dias", 30)
    contamination = meta.get("contamination", 0.02)

    time_from = client.days_ago_timestamp(dias)
    history = client.get_history(item_id, value_type, time_from)
    if not history:
        print(f"  [erro] sem historico Zabbix nos ultimos {dias} dias.")
        return False

    df_raw = history_to_dataframe(history)
    df_resampled = resample_series(df_raw, freq="5min")

    if payload.get("type") == "lstm":
        seq_len = payload.get("seq_len", 12)
        epochs = payload.get("epochs") or 20
        keras_model, scaler, threshold, _ = lstm_model.train(
            df_resampled, seq_len=seq_len, epochs=epochs, percentile=100 - contamination * 100,
        )
        lstm_model.save(keras_model, scaler, threshold, seq_len, 12, name, meta=meta, epochs=epochs)
    else:
        clf, _ = model_lib.train_model(df_resampled, contamination=contamination)
        model_lib.save_model(clf, name, meta=meta, model_type="single")

    print(f"  [ok] retreinado com {dias} dias de historico (contaminacao={contamination}).")
    return True


def _retrain_multi(client: ZabbixClient, name: str, payload: dict) -> bool:
    meta = payload.get("meta", {})
    items = meta.get("items", [])
    if not items:
        print(f"  [pular] modelo multi-metrica sem itens configurados.")
        return False

    dias = meta.get("dias", 30)
    contamination = meta.get("contamination", 0.02)

    time_from = client.days_ago_timestamp(dias)
    histories = {it["key"]: client.get_history(it["item_id"], it["value_type"], time_from) for it in items}
    df_combined = combine_histories(histories)
    if df_combined.empty:
        print(f"  [erro] sem dados combinados nos ultimos {dias} dias.")
        return False

    value_columns = [it["key"] for it in items]
    clf, _ = model_lib.train_multi_model(df_combined, value_columns, contamination=contamination)
    model_lib.save_model(clf, name, meta=meta, model_type="multi")

    print(f"  [ok] retreinado (multi-metrica, {len(items)} metricas) com {dias} dias de historico.")
    return True


def main():
    print(f"=== Retreino automatico - {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    try:
        client = get_client()
    except RuntimeError as e:
        print(f"[erro fatal] {e}")
        sys.exit(1)

    names = model_lib.list_saved_models()
    if not names:
        print("Nenhum modelo salvo para retreinar.")
        return

    ok_count, fail_count = 0, 0
    for name in names:
        print(f"Retreinando '{name}'...")
        payload = model_lib.load_model(name)
        if payload is None:
            print("  [erro] modelo nao encontrado ao carregar.")
            fail_count += 1
            continue
        try:
            if payload.get("type") == "multi":
                success = _retrain_multi(client, name, payload)
            else:
                success = _retrain_single(client, name, payload)
            ok_count += 1 if success else 0
            fail_count += 0 if success else 1
        except Exception as e:
            print(f"  [erro] {e}")
            fail_count += 1

    print(f"=== Concluido: {ok_count} retreinados, {fail_count} com problema ===")


if __name__ == "__main__":
    main()
