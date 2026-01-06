Files
- `generate_load.py` : Python script para gerar requisições GET/POST no app.

2. Rodando até 60s de teste:

```bash
python monitoring/loadgen/generate_load.py --host http://localhost:8000 --concurrency 30 --duration 60 --rate 200
```

Notes
- O script utiliza os métodos `/health`, `/health`, `/metrics`, `/stocks/${simbol}/status`,`/stocks/${simbol}/download`, `/predict` (POST) gerando chamadas a estes endpoint's e endpoint's invalidos para produzir erros, com isso gera dados de monitoramento no Grafana/Prometheus.
