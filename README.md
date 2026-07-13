# Detector-de-Anomalias-em-Infraestrutura

<img width="1263" height="297" alt="image" src="https://github.com/user-attachments/assets/0a47ced8-7df5-44db-bf3d-436b77c51ff2" />

<img width="1276" height="508" alt="image" src="https://github.com/user-attachments/assets/c7ea367c-5147-41ad-b188-66fcf64fe17c" />

<img width="1315" height="447" alt="image" src="https://github.com/user-attachments/assets/2d10fae3-98ca-457e-99d5-d987da55dfd0" />

<img width="1291" height="567" alt="image" src="https://github.com/user-attachments/assets/04273662-dfc5-43ce-983a-c1ff5c71ee3e" />

<img width="1291" height="449" alt="image" src="https://github.com/user-attachments/assets/6c267f91-237d-4c98-8c82-45eae8617642" />


# Detector de Anomalias em Infraestrutura (Zabbix)

Interface grafica (Streamlit) que conecta ao Zabbix, aprende o padrao normal
de uma metrica (CPU, rede, disco etc.) e destaca pontos fora do esperado,
usando Isolation Forest.

## Instalacao

```
cd anomaly-detector
pip install -r requirements.txt
```

## Como usar

1. Rode a interface:
   ```
   streamlit run app.py
   ```
   Isso abre uma pagina no navegador (geralmente http://localhost:8501).

2. **Aba "Conexao"**: informe a URL do seu Zabbix, escolha "API Token" (recomendado)
   ou "Usuario e senha", clique em **Testar conexao** e depois em **Conectar e salvar**.
   As credenciais ficam guardadas com seguranca no cofre do Windows (via `keyring`),
   nunca em texto puro.

3. **Aba "Treino"**:
   - Escolha o **host** e clique em "Carregar itens deste host".
   - Escolha o **item** (a metrica: CPU, disco, rede, etc.).
   - Ajuste quantos **dias de historico** usar para aprender o padrao normal
     (comece com 30).
   - Ajuste a **sensibilidade** (quanto maior, mais pontos serao marcados
     como anomalia).
   - Clique em **Buscar dados e treinar**. Vai aparecer um grafico com a
     serie e os pontos anomalos marcados em vermelho.
   - Se o resultado fizer sentido, de um nome ao modelo e clique em
     **Salvar modelo**.

4. **Aba "Monitoramento"**: lista os modelos ja treinados e salvos.

## Estrutura

```
anomaly-detector/
  app.py                  # interface Streamlit (ponto de entrada)
  src/
    zabbix_client.py      # comunicacao com a API do Zabbix
    credentials.py        # armazenamento seguro de credenciais (keyring)
    features.py           # transformacao da serie bruta em features
    model.py              # treino/salvamento/inferencia do Isolation Forest
  models/                 # modelos treinados (.joblib), criado automaticamente
  requirements.txt
```

## Proximos passos (evolucao futura)

- Script `monitor.py` para rodar em segundo plano, reavaliar um modelo salvo
  periodicamente e disparar alerta (Zabbix trap, e-mail, Telegram).
- Retreino automatico agendado (ex: 1x por semana).
