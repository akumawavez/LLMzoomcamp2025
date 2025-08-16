FROM docker.elastic.co/elasticsearch/elasticsearch:8.4.3

# Set environment variables
ENV discovery.type=single-node
ENV xpack.security.enabled=false

# Set JVM heap size (optional but useful for 4GB memory limit)
ENV ES_JAVA_OPTS="-Xms2g -Xmx2g"

# Expose ports (Elasticsearch HTTP & transport)
EXPOSE 9200 9300
