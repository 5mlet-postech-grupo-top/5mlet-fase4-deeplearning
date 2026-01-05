Kubernetes
-----------

Também incluí manifests simples em `k8s/` para deploy em cluster (ConfigMaps, Deployments, Services e PVCs):
Aplicar os manifests:

```bash
kubectl apply -f k8s/prometheus-configmap.yml
kubectl apply -f k8s/prometheus-deployment.yml
kubectl apply -f k8s/grafana-configmap.yml
kubectl apply -f k8s/grafana-deployment.yml
```
Atenção: estes manifests são um ponto de partida — para produção use Helm charts ou o Prometheus Operator (`kube-prometheus-stack`) e ajuste recursos, storage class e políticas de segurança.
Monitoramento simples com Prometheus + Grafana

Objetivo:
- Fornecer uma configuração mínima para coletar métricas da aplicação publicada em:
  https://fivemlet-fase4-deeplearning.onrender.com/

O que foi adicionado:
- `monitoring/prometheus/prometheus.yml` : configuração do Prometheus que faz scrape em `/metrics` no domínio da aplicação (HTTPS).
- `monitoring/docker-compose.yml` : composição para subir Prometheus e Grafana localmente para testar e visualizar métricas.
- `monitoring/grafana/provisioning` : provisionamento de datasource (Prometheus) e dashboards.
- `monitoring/grafana/dashboards/basic_metrics.json` : dashboard mínimo com taxa de requisições e latência.
- Atualização de `requirements.txt` com `prometheus-client`.
- Integração no `src/api.py` que expõe `/metrics` usando `prometheus-client` e atualiza métricas no middleware.

Como usar localmente (testar):
1. Instale dependências do Python (recomendado em virtualenv):

```bash
python -m pip install -r requirements.txt
```

2. Rode a aplicação localmente (exemplo):

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

3. Suba Prometheus e Grafana (na raiz `monitoring/`):

```bash
cd monitoring
docker-compose up -d
```

- Grafana: http://localhost:3000 (usuário: `admin`, senha: `admin`)
- Prometheus UI: http://localhost:9090

Observações para ambiente de produção (Render):
- O Prometheus configurado em `prometheus.yml` faz scrape direto no domínio `https://fivemlet-fase4-deeplearning.onrender.com/metrics`.
- Garanta que o endpoint `/metrics` esteja acessível publicamente pelo Prometheus (sem autenticação) ou proteja via rede e use autenticação mútua conforme necessário.

Próximos passos recomendados:
- Ajustar retenção e armazenamento do Prometheus para produção (filas, PVCs, Thanos/Cortex, etc.).
- Proteger endpoints e considerar autenticação/ACL para acesso a métricas sensíveis.
- Usar deploy via Kubernetes (adicionar manifests em `k8s/`) se preferir integrar ao cluster.

