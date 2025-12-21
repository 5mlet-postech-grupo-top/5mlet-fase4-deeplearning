# 📈 LSTM Stock Price Predictor (MLOps & Cloud Ready)

### Previsão de preços de ações com redes neurais LSTM, FastAPI, Treino Assíncrono e Persistência no Hugging Face.

Esta é a versão final do Tech Challenge (Fase 4). A solução foi evoluída para suportar ambientes de nuvem efêmeros (como o Render), utilizando o Hugging Face como repositório de modelos.

### 🌟 Novas Funcionalidades:

* ✔️ **MLOps:** Integração nativa com **Hugging Face Hub**.
* ✔️ **Persistência em Nuvem:** Modelos treinados são enviados automaticamente para a nuvem.
* ✔️ **Auto-Recovery:** Se o container reiniciar, a API baixa o modelo automaticamente antes de predizer.
* ✔️ **Docker Otimizado:** Configurado para produção (Render/K8s).
* ✔️ **Treino Assíncrono:** Não bloqueia a API durante o treinamento.

---

# 🚀 Fluxo de MLOps

1. **Download:** Usuário solicita download dos dados (salvos temporariamente).
2. **Treino:** API treina o modelo LSTM em background.
3. **Upload (Auto):** Ao finalizar, a API envia o modelo (`.h5`) e o scaler (`.pkl`) para o seu repositório no Hugging Face.
4. **Deploy/Reinício:** O servidor pode ser desligado ou reiniciado (dados locais são perdidos).
5. **Predição (Auto-Restore):** Ao solicitar uma previsão, se o modelo não estiver no disco, a API **baixa automaticamente do Hugging Face** e executa a inferência.

---

# ⚙️ Configuração Prévia (Obrigatório)

Para que a persistência funcione, você precisa de um token do Hugging Face.

1. Crie uma conta em [huggingface.co](https://huggingface.co/).
2. Crie um **Model Repository** (ex: `seu-usuario/tech-challenge-fase4`).
3. Gere um **Access Token** com permissão de `WRITE` (Settings > Access Tokens).

Defina as variáveis de ambiente no seu sistema ou no arquivo `.env` (se usar):

* `HF_TOKEN`: Seu token de escrita.
* `HF_REPO_ID`: O ID do repositório (ex: `joao/stock-lstm`).

---

# 🐳 Docker

O Dockerfile foi ajustado para rodar na estrutura de pastas `src/` e aceitar a porta dinâmica do Render.

### 1. Build da Imagem

```bash
docker build -t lstm-api .

```

### 2. Rodar o Container

Você **deve** passar as variáveis de ambiente para que o upload/download funcione:

```bash
docker run -p 8000:8000 \
  -e HF_TOKEN="seu_token_aqui" \
  -e HF_REPO_ID="seu-usuario/nome-do-repo" \
  lstm-api

```

Swagger disponível em: 👉 http://localhost:8000/docs

---

# 💻 Rodando Localmente (Sem Docker)

### 1. Instalar Dependências

Recomendado usar Python 3.10 ou 3.11.

```bash
pip install -r requirements.txt

```

### 2. Configurar Variáveis (Linux/Mac)

```bash
export HF_TOKEN="seu_token_aqui"
export HF_REPO_ID="seu-usuario/nome-do-repo"

```

*(No Windows Powershell use: `$env:HF_TOKEN=".."`)*

### 3. Executar a API

Entre na pasta `src` e inicie o servidor:

```bash
cd src
uvicorn api:app --reload --port 8000

```

Acesse: 👉 http://localhost:8000/docs

---

# 📡 Endpoints Principais

## 1️⃣ Download de Dados

Baixa dados do Yahoo Finance para o disco temporário.

`POST /stocks/{symbol}/download`

```bash
curl -X POST "http://localhost:8000/stocks/AAPL/download" \
     -H "Content-Type: application/json" \
     -d '{"start_date": "2018-01-01"}'

```

## 2️⃣ Treinar Modelo (Com Upload Automático)

Inicia o treino e, ao final, faz o upload para o Hugging Face.

`POST /stocks/{symbol}/train`

```bash
curl -X POST "http://localhost:8000/stocks/AAPL/train"

```

## 3️⃣ Consultar Status

Verifica se o modelo está pronto (localmente).

`GET /stocks/{symbol}/status`
### 📈 Endpoint de Monitoramento — /metrics-summary

## 4️⃣ Predição (Com Download Automático)

Se o modelo não estiver na máquina, a API busca no Hugging Face.

`POST /predict`

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "symbol": "AAPL",
           "last_n_days": 60,
           "n_days": 5
         }'

```

---

# ☁️ Deploy no Render

1. Crie um **Web Service** no Render conectado ao seu Git.
2. Runtime: **Docker**.
3. Adicione as **Environment Variables** no painel do Render:
* `HF_TOKEN`: (Seu token)
* `HF_REPO_ID`: (Seu repo)


4. O Render injetará automaticamente a variável `PORT`.

---

# 📊 Monitoramento

### Health Check

`GET /health`

### Métricas de Uso

`GET /metrics-summary`
Retorna uso de CPU, Memória e tempos de resposta da API.

---

# ✔️ Conclusão

Esta arquitetura resolve o problema de **"Cold Start"** e **Sistemas de Arquivos Efêmeros** em containers serverless. O modelo treinado hoje estará disponível amanhã, mesmo que o servidor seja destruído e recriado, garantindo uma pipeline de MLOps robusta.