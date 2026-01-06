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
```

Local vs Produção
------------------

- Local (docker-compose): o `docker-compose.yml` usa `prometheus/prometheus-local.yml`, que está configurado para fazer scrape em `host.docker.internal:8000` (útil quando você roda a aplicação com `uvicorn` no host). Para testar localmente:

```bash
cd monitoring
docker-compose up -d
```

Opções locais:
- Rodar a aplicação no host (uvicorn): o Prometheus local tenta `host.docker.internal:8000`.

```bash
# em outro terminal (na raiz do repo)
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

- Rodar a aplicação também via `docker-compose` (recomendado para reproduzir ambiente de container): o `docker-compose.yml` agora inclui um serviço `app` que sobe a aplicação (usa o `Dockerfile` do projeto). Para subir tudo:

```bash
cd monitoring
docker-compose up -d
```

Isso cria os serviços `app`, `prometheus` e `grafana`. Abra http://localhost:9090 (Prometheus) e http://localhost:3000 (Grafana).

Grafana fora do Docker
----------------------

Se você executar o Grafana localmente (não via Docker), defina a variável de ambiente `PROMETHEUS_URL` antes de iniciar o Grafana para que o provisioning aponte para o Prometheus correto. Exemplos:

- Se estiver usando Prometheus em `docker-compose` (localhost:9090):

```bash
export PROMETHEUS_URL=http://localhost:9090
# iniciar grafana (depende de como você instala/roda o grafana localmente)
```

- Se o Prometheus estiver remoto (produção), use:

```bash
export PROMETHEUS_URL=https://fivemlet-fase4-deeplearning.onrender.com
```

Nota: o provisioning do Grafana lê a variável `PROMETHEUS_URL` quando o Grafana inicia — altere a variável e reinicie o Grafana para aplicar a mudança.

- Produção: o arquivo `monitoring/prometheus/prometheus.yml` (mantido no repositório) está preparado para fazer scrape do endpoint público `https://fivemlet-fase4-deeplearning.onrender.com/metrics`. Em produção, use este `prometheus.yml` no servidor do Prometheus (não no `docker-compose` local).

Se preferir executar um Prometheus local que faça scrape da aplicação local no próprio container (por exemplo se a app também estiver em containers), ajuste `prometheus-local.yml` para apontar ao nome do serviço do container em vez de `host.docker.internal`.

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

