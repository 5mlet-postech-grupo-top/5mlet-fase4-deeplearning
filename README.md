# 📈 LSTM Stock Price Predictor (Versão Evoluída)
### Previsão de preços de ações com redes neurais LSTM, FastAPI, Treino Assíncrono e Suporte Multi-Ação

Esta versão evoluída da solução permite:

- ✔️ Download sob demanda de dados de qualquer ação  
- ✔️ Treinamento assíncrono por ação (sem timeout)  
- ✔️ Status completo de dados, modelo e treino  
- ✔️ Predição condicionada à existência de modelo treinado  
- ✔️ Janelas dinâmicas via `last_n_days`  
- ✔️ Monitoramento da API  
- ✔️ Documentação completa via Swagger  

---

# 🚀 Fluxo Completo

1. **Baixar dados da ação**  
2. **Consultar status**  
3. **Iniciar treino assíncrono**  
4. **Consultar progresso / métricas do treino**  
5. **Executar predição**  

---

# 📥 1) Download de Dados da Ação

### **Endpoint**
`POST /stocks/{symbol}/download`

Baixa dados desde 2018-01-01 até hoje por padrão.

### **curl**
```bash
curl -X POST "http://localhost:8000/stocks/AAPL/download"      -H "Content-Type: application/json"      -d '{"start_date": "2018-01-01"}'
```

### **Resposta**
```json
{
  "symbol": "AAPL",
  "rows": 1580,
  "data_path": ".../data/AAPL_history.csv"
}
```

---

# 🔍 2) Consultar Status da Ação

Mostra:

- se existem dados  
- se existe modelo treinado  
- se está pronto para predição  
- estado do treino (idle / running / success / error)  
- métricas do último treino  

### **Endpoint**
`GET /stocks/{symbol}/status`

### **curl**
```bash
curl "http://localhost:8000/stocks/AAPL/status"
```

### **Resposta**
```json
{
  "symbol": "AAPL",
  "has_data": true,
  "has_model": false,
  "ready_for_prediction": false,
  "training_status": "idle"
}
```

---

# 🧠 3) Treinar Modelo da Ação (Assíncrono)

Treina o modelo em background sem bloquear o cliente.

### **Endpoint**
`POST /stocks/{symbol}/train`

### **curl**
```bash
curl -X POST "http://localhost:8000/stocks/AAPL/train"
```

### **Resposta imediata**
```json
{
  "symbol": "AAPL",
  "started": true,
  "status_url": "http://localhost:8000/stocks/AAPL/status"
}
```

### **Exemplo de status durante o treino**
```json
{
  "symbol": "AAPL",
  "training_status": "running"
}
```

### **Exemplo após o término**
```json
{
  "symbol": "AAPL",
  "training_status": "success",
  "training_metrics": {
    "mae": 1.23,
    "rmse": 1.74,
    "mape": 1.82
  }
}
```

---

# 📊 4) Predição por Ação

Só funciona se o modelo estiver treinado.

### **Endpoint**
`POST /predict`

### **curl**
```bash
curl -X POST "http://localhost:8000/predict"      -H "Content-Type: application/json"      -d '{
           "symbol": "AAPL",
           "last_n_days": 120,
           "n_days": 3
         }'
```

### **Resposta**
```json
{
  "symbol": "AAPL",
  "predicted_prices": [195.31, 195.82, 196.44]
}
```

### ❗ Caso o modelo não exista:
```json
{
  "detail": {
    "message": "Não existe modelo treinado para AAPL.",
    "docs": "http://localhost:8000/docs"
  }
}
```

---

# 📉 5) Monitoramento da API

### **Health Check**

`GET /health`

```bash
curl http://localhost:8000/health
```

---

### **Resumo de métricas**

`GET /metrics-summary`

```bash
curl http://localhost:8000/metrics-summary
```

**Exemplo:**
```json
{
  "total_requests": 57,
  "total_errors": 0,
  "avg_response_time_ms": 14.82,
  "max_response_time_ms": 51.44,
  "cpu_percent": 9.1,
  "memory_rss_mb": 143.2
}
```

---

# 🐳 6) Docker

### Build
```bash
docker build -t lstm-api .
```

### Executar
```bash
docker run -p 8000:8000 lstm-api
```

Swagger:
👉 http://localhost:8000/docs

---

# ✔️ Conclusão

Esta versão do produto entrega:

- Download de dados sob demanda  
- Treino assíncrono seguro  
- Status avançado por ativo  
- Predição multi-ação  
- Monitoramento real  
- API pronta para produção  

Ideal para pipelines avançadas e evolução contínua do modelo.
