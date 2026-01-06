FROM grafana/grafana:9.5.0

# Copy provisioning templates and entrypoint
COPY provisioning /provisioning.template
COPY docker-entrypoint.sh /docker-entrypoint.sh

RUN chmod +x /docker-entrypoint.sh \
    && mkdir -p /etc/grafana/provisioning/datasources /etc/grafana/provisioning/dashboards \
    && if [ -d /provisioning.template/datasources ]; then \
         cp /provisioning.template/datasources/* /etc/grafana/provisioning/datasources/ || true; \
       fi \
    && if [ -d /provisioning.template/dashboards ]; then \
         cp /provisioning.template/dashboards/* /etc/grafana/provisioning/dashboards/ || true; \
       fi \
    && if [ -f /etc/grafana/provisioning/datasources/datasource.yml ]; then \
         mv /etc/grafana/provisioning/datasources/datasource.yml /etc/grafana/provisioning/datasources/datasource.yml.template || true; \
       fi

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/run.sh"]
