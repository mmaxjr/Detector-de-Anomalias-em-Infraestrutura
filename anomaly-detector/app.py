"""
Detector de Anomalias em Metricas de Infraestrutura (Zabbix)
--------------------------------------------------------------
Interface grafica (Streamlit) para conectar ao Zabbix, treinar modelos de
deteccao de anomalias (Isolation Forest single/multi-metrica ou LSTM
autoencoder) e monitorar os resultados - tudo sem precisar escrever codigo.

Para rodar:
    streamlit run app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.zabbix_client import ZabbixClient, ZabbixAPIError
from src.credentials import save_connection, load_connection, clear_connection
from src.features import history_to_dataframe, resample_series
from src.multi_features import combine_histories
from src import model as model_lib
from src import lstm_model
from src import monitor

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

st.set_page_config(page_title="Detector de Anomalias - Zabbix", layout="wide")

STATUS_COLORS = {"green": "#1f8f3d", "yellow": "#b8860b", "red": "#b23b3b", "gray": "#555555"}
STATUS_LABELS = {"green": "Normal", "yellow": "Atencao", "red": "Anomalia", "gray": "Sem dados"}

# ---------------------------------------------------------------------
# Estado da sessao
# ---------------------------------------------------------------------
for key, default in [
    ("client", None), ("hosts", []), ("zbx_items", []),
    ("last_result", None), ("last_value_columns", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("Detector de Anomalias em Infraestrutura")
st.caption("Conecta no Zabbix, aprende o padrao normal de uma metrica e aponta o que fugiu do esperado.")

tab_conexao, tab_treino, tab_monitor, tab_dashboard = st.tabs(
    ["1. Conexao", "2. Treino", "3. Monitoramento", "4. Dashboard"]
)


def plot_result(df_result: pd.DataFrame, value_columns: list, height: int = 400, key: str = None):
    """Plota a(s) serie(s) e marca os pontos de anomalia (comum entre single/multi/lstm)."""
    fig = go.Figure()
    palette = ["#4C78A8", "#72B7B2", "#E45756", "#54A24B", "#EECA3B", "#B279A2"]
    for i, col in enumerate(value_columns):
        if col in df_result.columns:
            fig.add_trace(go.Scatter(
                x=df_result["timestamp"], y=df_result[col],
                mode="lines", name=col, line=dict(color=palette[i % len(palette)]),
            ))
    anomalies = df_result[df_result["is_anomaly"]]
    y_ref = value_columns[0] if value_columns and value_columns[0] in df_result.columns else "value"
    if y_ref in df_result.columns:
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"], y=anomalies[y_ref],
            mode="markers", name="Anomalia",
            marker=dict(color="red", size=9, symbol="x"),
        ))
    fig.update_layout(height=height, margin=dict(t=30, b=30), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True, key=key)


# =======================================================================
# ABA 1 - CONEXAO
# =======================================================================
with tab_conexao:
    st.subheader("Conexao com o Zabbix")

    saved = load_connection()
    if saved and st.session_state.client is None:
        st.info("Ha uma conexao salva. Clique em 'Conectar' para usa-la, ou edite os campos abaixo.")

    url = st.text_input(
        "URL do Zabbix (ex: http://192.168.0.10/zabbix)",
        value=(saved or {}).get("url", ""),
    )

    auth_mode = st.radio(
        "Metodo de autenticacao",
        options=["token", "senha"],
        format_func=lambda x: "API Token (recomendado)" if x == "token" else "Usuario e senha",
        index=0 if (saved or {}).get("auth_mode", "token") == "token" else 1,
        horizontal=True,
    )

    token = user = password = None
    if auth_mode == "token":
        token = st.text_input("API Token", value=(saved or {}).get("token", ""), type="password")
    else:
        col1, col2 = st.columns(2)
        with col1:
            user = st.text_input("Usuario", value=(saved or {}).get("user", ""))
        with col2:
            password = st.text_input("Senha", value=(saved or {}).get("password", ""), type="password")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        testar = st.button("Testar conexao", use_container_width=True)
    with col_b:
        conectar = st.button("Conectar e salvar", type="primary", use_container_width=True)
    with col_c:
        esquecer = st.button("Esquecer credenciais salvas", use_container_width=True)

    if esquecer:
        clear_connection()
        st.session_state.client = None
        st.success("Credenciais removidas.")

    if testar or conectar:
        try:
            client = ZabbixClient(url=url, token=token or None, user=user or None, password=password or None)
            version = client.test_connection()
            st.success(f"Conectado com sucesso! Versao da API do Zabbix: {version}")

            if conectar:
                save_connection(url=url, auth_mode=auth_mode, token=token, user=user, password=password)
                st.session_state.client = client
                st.session_state.hosts = client.get_hosts()
                st.success(f"{len(st.session_state.hosts)} hosts carregados. Va para a aba 'Treino'.")
        except ZabbixAPIError as e:
            st.error(f"Erro na API do Zabbix: {e}")
        except Exception as e:
            st.error(f"Nao foi possivel conectar: {e}")

# =======================================================================
# ABA 2 - TREINO
# =======================================================================
with tab_treino:
    st.subheader("Treinar modelo de anomalias")

    if st.session_state.client is None:
        st.warning("Conecte-se ao Zabbix na aba 'Conexao' primeiro.")
    else:
        hosts = st.session_state.hosts
        host_options = {f"{h['name']} ({h['host']})": h["hostid"] for h in hosts}
        host_label = st.selectbox("Host", options=list(host_options.keys()))
        host_id = host_options.get(host_label)

        tipo_modelo = st.radio(
            "Tipo de modelo",
            options=["single", "multi", "lstm"],
            format_func=lambda x: {
                "single": "Isolation Forest - 1 metrica (simples)",
                "multi": "Isolation Forest - varias metricas correlacionadas",
                "lstm": "LSTM Autoencoder - 1 metrica (avancado, capta padroes temporais)",
            }[x],
        )

        if host_id:
            if st.button("Carregar itens deste host"):
                st.session_state.zbx_items = st.session_state.client.get_items(host_id)

            if st.session_state.zbx_items:
                item_options = {f"{i['name']} [{i['key_']}]": i for i in st.session_state.zbx_items}

                # -----------------------------------------------------
                # Selecao de item(s) - unica para single/lstm, multipla para multi
                # -----------------------------------------------------
                if tipo_modelo == "multi":
                    labels_selecionados = st.multiselect(
                        "Metricas (selecione 2 ou mais do mesmo host)",
                        options=list(item_options.keys()),
                    )
                    itens_selecionados = [item_options[l] for l in labels_selecionados]
                else:
                    item_label = st.selectbox("Item (metrica)", options=list(item_options.keys()))
                    itens_selecionados = [item_options[item_label]]

                col1, col2 = st.columns(2)
                with col1:
                    dias = st.slider("Historico para treino (dias)", 3, 90, 30)
                with col2:
                    contamination = st.slider(
                        "Sensibilidade (% de pontos esperados como anomalia)",
                        min_value=0.5, max_value=10.0, value=2.0, step=0.5,
                    ) / 100.0

                seq_len = epochs = None
                if tipo_modelo == "lstm":
                    with st.expander("Parametros avancados do LSTM"):
                        seq_len = st.slider("Tamanho da janela (pontos consecutivos)", 6, 48, 12)
                        epochs = st.slider("Epocas de treino", 5, 100, 20)

                pode_treinar = (
                    (tipo_modelo != "multi" and len(itens_selecionados) == 1)
                    or (tipo_modelo == "multi" and len(itens_selecionados) >= 2)
                )
                if tipo_modelo == "multi" and len(itens_selecionados) < 2:
                    st.info("Selecione pelo menos 2 metricas para treinar um modelo correlacionado.")

                if pode_treinar and st.button("Buscar dados e treinar", type="primary"):
                    time_from = st.session_state.client.days_ago_timestamp(dias)

                    if tipo_modelo == "multi":
                        with st.spinner("Buscando historico das metricas no Zabbix..."):
                            histories = {
                                it["key_"]: st.session_state.client.get_history(it["itemid"], it["value_type"], time_from)
                                for it in itens_selecionados
                            }
                        df_combined = combine_histories(histories)
                        if df_combined.empty:
                            st.error("Nao ha dados historicos suficientes/alinhados para essas metricas nesse periodo.")
                        else:
                            value_columns = [it["key_"] for it in itens_selecionados]
                            with st.spinner("Treinando modelo multi-metrica..."):
                                clf, df_result = model_lib.train_multi_model(
                                    df_combined, value_columns, contamination=contamination
                                )
                            st.session_state.last_result = df_result
                            st.session_state.last_value_columns = value_columns
                            st.session_state.last_model = clf
                            st.session_state.last_model_type = "multi"
                            st.session_state.last_items = itens_selecionados
                            st.session_state.last_host_label = host_label

                    elif tipo_modelo == "lstm":
                        item = itens_selecionados[0]
                        with st.spinner("Buscando historico no Zabbix..."):
                            history = st.session_state.client.get_history(item["itemid"], item["value_type"], time_from)
                        if not history:
                            st.error("Nenhum dado historico encontrado nesse periodo para esse item.")
                        else:
                            df_raw = history_to_dataframe(history)
                            df_resampled = resample_series(df_raw, freq="5min")
                            try:
                                with st.spinner("Treinando LSTM (pode levar alguns minutos)..."):
                                    keras_model, scaler, threshold, df_result = lstm_model.train(
                                        df_resampled, seq_len=seq_len, epochs=epochs,
                                        percentile=100 - contamination * 100,
                                    )
                                st.session_state.last_result = df_result
                                st.session_state.last_value_columns = ["value"]
                                st.session_state.last_model = (keras_model, scaler, threshold, seq_len)
                                st.session_state.last_model_type = "lstm"
                                st.session_state.last_item = item
                                st.session_state.last_host_label = host_label
                            except ValueError as e:
                                st.error(str(e))

                    else:  # single
                        item = itens_selecionados[0]
                        with st.spinner("Buscando historico no Zabbix..."):
                            history = st.session_state.client.get_history(item["itemid"], item["value_type"], time_from)
                        if not history:
                            st.error("Nenhum dado historico encontrado nesse periodo para esse item.")
                        else:
                            df_raw = history_to_dataframe(history)
                            df_resampled = resample_series(df_raw, freq="5min")
                            with st.spinner("Treinando modelo..."):
                                clf, df_result = model_lib.train_model(df_resampled, contamination=contamination)
                            st.session_state.last_result = df_result
                            st.session_state.last_value_columns = ["value"]
                            st.session_state.last_model = clf
                            st.session_state.last_model_type = "single"
                            st.session_state.last_item = item
                            st.session_state.last_host_label = host_label

                # -----------------------------------------------------
                # Resultado do treino + salvar
                # -----------------------------------------------------
                if st.session_state.get("last_result") is not None:
                    df_result = st.session_state.last_result
                    value_columns = st.session_state.get("last_value_columns", ["value"])
                    n_anom = int(df_result["is_anomaly"].sum())
                    st.markdown(f"**{n_anom} pontos marcados como anomalia** de {len(df_result)} no periodo.")

                    plot_result(df_result, value_columns, height=450, key="treino_chart")

                    st.caption("Ajuste a sensibilidade acima e clique em 'Buscar dados e treinar' novamente para recalibrar.")

                    _last_type = st.session_state.get("last_model_type", "modelo")
                    if _last_type == "multi":
                        _metric_part = "_".join(it["key_"] for it in st.session_state.get("last_items", []))[:60]
                    else:
                        _metric_part = st.session_state.get("last_item", {}).get("key_", "metrica")
                    default_name = f"{host_label}__{_metric_part}__{_last_type}"
                    model_name = st.text_input(
                        "Nome para salvar este modelo (edite como quiser)", value=default_name,
                        help="Esse texto e livre - mude para o que preferir antes de salvar.",
                    )

                    if st.button("Salvar modelo", type="primary"):
                        last_type = st.session_state.last_model_type

                        if last_type == "multi":
                            meta = {
                                "host": host_label,
                                "host_id": host_id,
                                "dias": dias,
                                "contamination": contamination,
                                "items": [
                                    {"key": it["key_"], "name": it["name"], "item_id": it["itemid"], "value_type": it["value_type"]}
                                    for it in st.session_state.last_items
                                ],
                            }
                            model_lib.save_model(st.session_state.last_model, model_name, meta=meta, model_type="multi")

                        elif last_type == "lstm":
                            keras_model, scaler, threshold, seql = st.session_state.last_model
                            item = st.session_state.last_item
                            meta = {
                                "host": host_label, "host_id": host_id,
                                "dias": dias, "contamination": contamination,
                                "item": item["name"], "key": item["key_"],
                                "item_id": item["itemid"], "value_type": item["value_type"],
                            }
                            lstm_model.save(keras_model, scaler, threshold, seql, 12, model_name, meta=meta, epochs=epochs)

                        else:  # single
                            item = st.session_state.last_item
                            meta = {
                                "host": host_label, "host_id": host_id,
                                "dias": dias, "contamination": contamination,
                                "item": item["name"], "key": item["key_"],
                                "item_id": item["itemid"], "value_type": item["value_type"],
                            }
                            model_lib.save_model(st.session_state.last_model, model_name, meta=meta, model_type="single")

                        st.success(f"Modelo salvo como '{model_name}'. Confira nas abas 'Monitoramento' e 'Dashboard'.")

# =======================================================================
# ABA 3 - MONITORAMENTO
# =======================================================================
with tab_monitor:
    st.subheader("Modelos salvos")

    _agora_utc = pd.Timestamp.utcnow()
    _agora_local = pd.Timestamp.now()
    st.caption(
        f"Horario que este app esta usando como 'agora': {_agora_local.strftime('%Y-%m-%d %H:%M:%S')} (local) "
        f"/ {_agora_utc.strftime('%Y-%m-%d %H:%M:%S')} (UTC). Compare com o relogio do Zabbix e da sua maquina."
    )

    if st.session_state.client is None:
        st.warning("Conecte-se ao Zabbix na aba 'Conexao' para poder verificar os modelos com dados atuais.")

    col_auto1, col_auto2 = st.columns([1, 2])
    with col_auto1:
        auto_refresh = st.checkbox("Atualizar automaticamente", value=False, key="auto_refresh_monitor")
    with col_auto2:
        intervalo_min = st.slider("A cada quantos minutos", 1, 30, 5, disabled=not auto_refresh, key="intervalo_monitor")

    if auto_refresh:
        if st_autorefresh is not None:
            st_autorefresh(interval=intervalo_min * 60 * 1000, key="monitor_autorefresh")
            st.caption(f"Atualizando sozinho a cada {intervalo_min} min.")
        else:
            st.caption("Instale streamlit-autorefresh (ja no requirements.txt) para atualizacao automatica.")

    saved_models = model_lib.list_saved_models()
    if not saved_models:
        st.info("Nenhum modelo treinado ainda. Va para a aba 'Treino'.")
    elif st.session_state.client is not None:
        for name in saved_models:
            payload = model_lib.load_model(name)
            meta = payload.get("meta", {}) if payload else {}
            with st.expander(f"📦 {name}", expanded=True):
                st.write(f"**Tipo:** {payload.get('type', 'single') if payload else '-'}")
                if payload and payload.get("type") == "multi":
                    itens_txt = ", ".join(it["key"] for it in meta.get("items", []))
                    st.write(f"**Host:** {meta.get('host', '-')} — **Metricas:** {itens_txt}")
                else:
                    st.write(f"**Host:** {meta.get('host', '-')}")
                    st.write(f"**Item:** {meta.get('item', '-')} ({meta.get('key', '-')})")

                col_r1, col_r2 = st.columns([2, 1])
                with col_r1:
                    novo_nome = st.text_input("Renomear modelo para:", value=name, key=f"rename_input_{name}")
                    if st.button("Renomear", key=f"rename_btn_{name}"):
                        if novo_nome and novo_nome != name:
                            ok_rename, msg_rename = model_lib.rename_model(name, novo_nome)
                            if ok_rename:
                                st.success(msg_rename)
                                st.rerun()
                            else:
                                st.error(msg_rename)
                with col_r2:
                    confirmar_del = st.checkbox("Confirmar exclusao", key=f"confirm_del_{name}")
                    if st.button("🗑️ Excluir modelo", key=f"del_btn_{name}", disabled=not confirmar_del):
                        model_lib.delete_model(name)
                        st.success(f"Modelo '{name}' excluido.")
                        st.rerun()

                st.divider()

                horas = st.slider("Ver dados das ultimas N horas", 1, 72, 6, key=f"horas_{name}")
                result = monitor.check_model(st.session_state.client, name, hours=horas)
                agora = pd.Timestamp.now().strftime("%H:%M:%S")

                status = result["status"]
                if status == "red":
                    st.error(f"⚠️ {result['message']} — verificado as {agora}")
                elif status == "yellow":
                    st.warning(f"{result['message']} — verificado as {agora}")
                elif status == "green":
                    st.success(f"{result['message']} — verificado as {agora}")
                else:
                    st.info(result["message"])

                if result["df"] is not None:
                    plot_result(result["df"], result["value_columns"], height=350, key=f"chart_{name}")

# =======================================================================
# ABA 4 - DASHBOARD (mapa de calor)
# =======================================================================
with tab_dashboard:
    st.subheader("Visao geral de todos os modelos")

    if st.session_state.client is None:
        st.warning("Conecte-se ao Zabbix na aba 'Conexao' para ver o dashboard.")
    else:
        col_auto1, col_auto2, col_auto3 = st.columns([1, 1, 2])
        with col_auto1:
            dash_auto = st.checkbox("Atualizar automaticamente", value=False, key="auto_refresh_dashboard")
        with col_auto2:
            dash_intervalo = st.slider("A cada (min)", 1, 30, 5, disabled=not dash_auto, key="intervalo_dashboard")
        with col_auto3:
            dash_horas = st.slider("Janela de verificacao (horas)", 1, 168, 24, key="dashboard_horas")

        if dash_auto and st_autorefresh is not None:
            st_autorefresh(interval=dash_intervalo * 60 * 1000, key="dashboard_autorefresh")

        saved_models = model_lib.list_saved_models()
        if not saved_models:
            st.info("Nenhum modelo treinado ainda. Va para a aba 'Treino'.")
        else:
            n_cols = 4
            cols = st.columns(n_cols)
            for idx, name in enumerate(saved_models):
                result = monitor.check_model(st.session_state.client, name, hours=dash_horas)
                color = STATUS_COLORS.get(result["status"], "#555555")
                label = STATUS_LABELS.get(result["status"], "Desconhecido")

                card_html = f"""
                <div style="background-color:{color}22; border:1px solid {color};
                            border-radius:8px; padding:12px; margin-bottom:12px;">
                    <div style="font-weight:600; font-size:0.95em; margin-bottom:4px;">{name}</div>
                    <div style="display:inline-block; background-color:{color}; color:white;
                                border-radius:4px; padding:2px 8px; font-size:0.8em;">{label}</div>
                    <div style="font-size:0.8em; margin-top:6px; opacity:0.85;">{result['message']}</div>
                </div>
                """
                with cols[idx % n_cols]:
                    st.markdown(card_html, unsafe_allow_html=True)

            st.caption(
                "Verde = normal, amarelo = anomalia recente mas ponto atual ok, "
                "vermelho = ponto atual fora do padrao, cinza = sem dados/erro."
            )
