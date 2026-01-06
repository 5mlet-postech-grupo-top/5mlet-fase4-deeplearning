FROM python:3.11-slim

WORKDIR /app

# Variáveis de ambiente para evitar arquivos .pyc e logs presos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Garante que huggingface_hub e tensorflow-cpu estejam no requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código fonte para a pasta src
COPY src/ ./src/

# Cria as pastas de dados e modelos DENTRO de src para facilitar o caminho relativo do script
# Mesmo que vazio, o diretório precisa existir para o script não falhar ao iniciar
RUN mkdir -p src/data src/models

# Define o diretório de trabalho onde o api.py está
WORKDIR /app/src

# O Render injeta a variável PORT. O comando deve usar sh -c para ler a variável.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]