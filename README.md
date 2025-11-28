# 📈 LSTM Stock Price Predictor  
### Previsão de preços de ações com redes neurais LSTM + FastAPI + Docker

Este projeto implementa uma pipeline completa para previsão de preços de ações utilizando redes neurais **LSTM**, incluindo:

- ✔️ Coleta de dados históricos  
- ✔️ Pré-processamento  
- ✔️ Treinamento LSTM  
- ✔️ Avaliação (MAE, RMSE, MAPE)  
- ✔️ Salvamento do modelo  
- ✔️ API REST com FastAPI  
- ✔️ Deploy com Docker  

---

## 📥 Coleta de Dados

Use o script `download_data.py` para baixar dados históricos de ações via **yfinance**.

### Executar:

```bash
cd src
python download_data.py
```

Um arquivo CSV será gerado contendo os dados históricos da ação escolhida.

---

## 🤖 Treinamento do Modelo LSTM

Execute o script `train_lstm.py` para:

- converter e limpar dados  
- criar janelas temporais  
- treinar a rede LSTM  
- validar o modelo  
- calcular métricas  
- gerar artefatos do modelo  

### Executar:

```bash
python train_lstm.py
```

O treinamento irá gerar:

- `lstm_model.h5`  
- `scaler.pkl`  

---

## 📊 Métricas de Avaliação

O modelo é avaliado usando:

- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Square Error  
- **MAPE** – Mean Absolute Percentage Error  

Exemplo de saída:

```
MAE: 1.23
RMSE: 1.74
MAPE: 1.82%
```

---

## 🌐 API REST com FastAPI

O arquivo `api.py` cria uma API que recebe preços históricos e retorna previsões.

### Executar a API:

```bash
uvicorn api:app --reload --port 8000
```

Acesse a documentação interativa em:

👉 http://localhost:8000/docs

---

### 🔮 Exemplo de requisição

POST `/predict`:

```json
{
  "prices": [
  171.52, 172.10, 173.04, 172.88, 173.55, 174.02, 174.44, 173.98, 174.62, 175.10,
  175.48, 176.02, 175.66, 176.31, 176.89, 177.42, 176.95, 177.60, 178.14, 178.72,
  179.05, 178.66, 179.20, 179.88, 180.33, 180.92, 181.35, 181.82, 182.40, 181.94,
  182.55, 182.98, 183.52, 184.00, 184.44, 184.92, 185.30, 185.82, 186.24, 186.70,
  187.18, 187.66, 188.14, 188.60, 189.02, 189.55, 189.98, 190.40, 190.88, 191.35,
  191.82, 192.20, 192.72, 193.10, 193.58, 194.05, 194.48, 194.92, 194.48, 194.92
],
  "n_days": 3
}
```

### Resposta:

```json
{
  "predicted_prices": [174.82, 175.13, 175.44]
}
```

---

## 🐳 Deploy com Docker

### Build:

```bash
docker build -t lstm-api .
```

### Executar:

```bash
docker run -p 8000:8000 lstm-api
```

---

## 📝 Instalando Dependências

```bash
pip install -r requirements.txt
```

---

## 📹 O que mostrar no vídeo da entrega

- coleta dos dados  
- treinamento do modelo  
- métricas de avaliação  
- arquivos exportados  
- API rodando  
- requisição funcionando  
- execução via Docker  

---

## 🌟 Melhorias Futuras

- uso de indicadores técnicos (RSI, MACD, médias móveis)  
- re-treino automático periódico  
- deploy em AWS (ECS, Lambda, Beanstalk)  
- monitoramento com Prometheus e Grafana  
- criação de dashboard com Streamlit  

---

## ✔️ Conclusão

Este projeto entrega toda a pipeline exigida no Tech Challenge: coleta, modelo LSTM, avaliação, API e Docker — tudo documentado e pronto para uso.
