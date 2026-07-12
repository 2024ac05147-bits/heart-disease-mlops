# Monitoring Stack

This directory provides a local Prometheus and Grafana monitoring stack for
the Heart Disease Prediction API.

## Components

- Heart Disease FastAPI application
- Prometheus metrics collection
- Grafana dashboard
- Persistent Prometheus and Grafana storage

## Monitored metrics

- API availability
- Total API requests
- Request rate by endpoint and HTTP status
- Total predictions
- Predictions by model output class
- Model prediction latency
- Model inference errors

## Start the stack

From the repository root:

```bash
docker compose \
  -f deployment/monitoring/docker-compose.yml \
  up --build -d
```

## Verify containers

```bash
docker compose \
  -f deployment/monitoring/docker-compose.yml \
  ps
```

## Access services

- FastAPI Swagger: http://127.0.0.1:8000/docs
- FastAPI metrics: http://127.0.0.1:8000/metrics
- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000

Default local Grafana credentials:

```text
Username: admin
Password: admin
```

Change the password for any non-local environment.

## Generate prediction traffic

```bash
curl \
  --request POST \
  --header "Content-Type: application/json" \
  --data @sample_request.json \
  http://127.0.0.1:8000/predict
```

Repeat the command several times so that Prometheus and Grafana display
request, prediction, and latency data.

## Prometheus verification

Open:

```text
http://127.0.0.1:9090/targets
```

The `heart-disease-api` target should display as `UP`.

Example PromQL queries:

```promql
up{job="heart-disease-api"}
```

```promql
sum(heart_api_requests_total)
```

```promql
sum by (prediction_class) (
  heart_model_predictions_total
)
```

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(heart_model_prediction_seconds_bucket[5m])
  )
)
```

## Grafana dashboard

The dashboard is automatically provisioned under:

```text
Dashboards → Heart Disease MLOps
```

Dashboard name:

```text
Heart Disease MLOps Monitoring
```

## Stop the stack

```bash
docker compose \
  -f deployment/monitoring/docker-compose.yml \
  down
```

To also remove monitoring data:

```bash
docker compose \
  -f deployment/monitoring/docker-compose.yml \
  down -v
```