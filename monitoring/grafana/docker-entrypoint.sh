#!/bin/sh
set -e

TEMPLATE_DIR=/etc/grafana/provisioning/datasources
TEMPLATE_FILE=${TEMPLATE_DIR}/datasource.yml.template
OUT_FILE=${TEMPLATE_DIR}/datasource.yml

if [ -f "$TEMPLATE_FILE" ]; then
  URL="${PROMETHEUS_URL:-http://prometheus:9090}"
  esc_url=$(printf '%s' "$URL" | sed -e 's/[\/&]/\\&/g')
  sed "s/\${PROMETHEUS_URL}/$esc_url/g" "$TEMPLATE_FILE" > "$OUT_FILE"
fi

exec /run.sh "$@"
