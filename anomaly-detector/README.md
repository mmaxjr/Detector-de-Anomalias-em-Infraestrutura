# Detector de Anomalias em Infraestrutura (Zabbix)

Interface grafica (Streamlit) que conecta ao Zabbix, aprende o padrao normal
de metricas (CPU, rede, disco, etc.) e destaca o que fugiu do esperado.
Suporta tres tipos de modelo: Isolation Forest simples (1 metrica),
Isolation Forest multi-metrica (correlacao entre metricas) e LSTM
Autoencoder (padroes temporais mais sutis).

## Instalacao

```
cd anomaly-detector
pip install -r requirements.txt
```

O TensorFlow (usado pelo LSTM) e um download grande (~350MB). Se a instalacao
cair no meio, tente:
```
pip install --timeout 120 --retries 5 tensorflow
```

## Como usar

1. Rode a interface:
   ```
   streamlit run app.py
   ```
   Abre uma pagina no navegador (geralmente http://localhost:8501).

2. **Aba "Conexao"**: informe a URL do Zabbix, escolha "API Token" (recomendado)
   ou "Usuario e senha", clique em **Testar conexao** e depois **Conectar e salvar**.
   As credenciais ficam guardadas com seguranca no cofre do Windows (via `keyring`).

3. **Aba "Treino"**:
   - Escolha o **host** e clique em "Carregar itens deste host".
   - Escolha o **tipo de modelo**:
     - *Isolation Forest - 1 metrica*: caso simples, uma metrica por vez.
     - *Isolation Forest - varias metricas correlacionadas*: selecione 2+
       metricas do mesmo host para detectar quando elas saem do padrao
       juntas (ex: CPU + rede + disco).
     - *LSTM Autoencoder*: mais pesado de treinar, mas capta padroes
       temporais (ex: pico normal de dia, anomalo de madrugada).
   - Ajuste **dias de historico** e **sensibilidade**, clique em
     **Buscar dados e treinar** e confira o grafico.
   - De um nome ao modelo (o nome sugerido ja inclui host + metrica + tipo)
     e clique em **Salvar modelo**.

4. **Aba "Monitoramento"**: lista os modelos salvos, roda a verificacao
   automaticamente com dados recentes do Zabbix (sem precisar clicar em
   nada), com opcao de atualizacao automatica periodica. Cada modelo pode
   ser **renomeado** ou **excluido** direto pela interface.

5. **Aba "Dashboard"**: visao geral tipo mapa de calor de todos os modelos
   salvos de uma vez (verde = normal, amarelo = anomalia recente, vermelho
   = ponto atual fora do padrao, cinza = sem dados/erro).

### Se aparecer "Sem dados no periodo"

O app mostra automaticamente a ultima coleta real do item segundo o
proprio Zabbix. Se essa data for recente mas mesmo assim "fora da janela",
o problema costuma ser: item com intervalo de coleta longo (aumente a
janela de horas), item/host que parou de coletar (confira em
"Latest data" no Zabbix), ou relogio dessincronizado entre esta maquina e
o servidor Zabbix.

## Retreino automatico (agendado)

Os modelos ficam desatualizados conforme o padrao normal da infraestrutura
muda. Em vez de retreinar manualmente pela aba Treino, use o script
`retrain_all.py`, que retreina TODOS os modelos salvos usando os mesmos
parametros do treino original (dias de historico, sensibilidade, epocas do
LSTM - tudo isso fica salvo automaticamente com o modelo).

Teste manual:
```
python retrain_all.py
```

### Agendar no Windows (recomendado: 1x por semana)

1. Abra o **Agendador de Tarefas** do Windows (pesquise "Task Scheduler").
2. **Criar Tarefa Basica...** > de um nome (ex: "Retreino Detector Anomalias").
3. **Disparador**: Semanalmente, escolha dia e horario (ex: domingo, 03:00).
4. **Acao**: Iniciar um programa.
   - Programa/script: caminho do `python.exe` do seu ambiente
     (ex: `C:\Users\SeuUsuario\AppData\Local\Programs\Python\Python311\python.exe`)
   - Argumentos: `retrain_all.py`
   - Iniciar em: o caminho completo da pasta `anomaly-detector`
     (ex: `D:\MAX\SUPORTE-SYMA\anomaly-detector`)
5. Finalize. A tarefa vai rodar sozinha no horario configurado, mesmo com
   o Streamlit fechado - so precisa ter feito **Conectar e salvar** pelo
   menos uma vez na aba Conexao para o script ter credenciais salvas.

Dica: rode `python retrain_all.py > retrain_log.txt 2>&1` como argumento
completo se quiser guardar um log de cada execucao.

## Estrutura

```
anomaly-detector/
  app.py                  # interface Streamlit (ponto de entrada)
  retrain_all.py          # script standalone de retreino agendado
  src/
    zabbix_client.py      # comunicacao com a API do Zabbix
    credentials.py        # armazenamento seguro de credenciais (keyring)
    features.py           # transformacao da serie bruta em features (1 metrica)
    multi_features.py     # combinacao de varias metricas (correlacao)
    model.py              # treino/salvamento/inferencia (Isolation Forest)
    lstm_model.py          # treino/salvamento/inferencia (LSTM Autoencoder)
    monitor.py             # verificacao unificada usada por Monitoramento e Dashboard
  models/                 # modelos treinados (.joblib + pastas _lstm), criado automaticamente
  requirements.txt
```

## Proximos passos (evolucao futura)

- Alertas ativos (Telegram, e-mail, webhook) quando uma anomalia e detectada,
  em vez de precisar abrir o Dashboard.
- Checagem dedicada de "host parou de coletar" (usando lastclock), separada
  do modelo de ML.
- Historico de anomalias em banco (SQLite) para relatorios de tendencia.
- Enviar deteccoes de volta pro Zabbix como trap/item, aproveitando o
  sistema de alertas nativo dele.
