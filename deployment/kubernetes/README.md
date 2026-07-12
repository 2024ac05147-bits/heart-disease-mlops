# Kubernetes Deployment

This directory contains portable Kubernetes manifests for deploying the
Heart Disease Prediction API to Minikube, Docker Desktop Kubernetes, kind,
or a standard cloud Kubernetes cluster.

## Included resources

- Dedicated `heart-disease` namespace
- Deployment with two API replicas
- Rolling update strategy
- Startup, readiness, and liveness probes
- CPU and memory requests and limits
- Non-root container security settings
- LoadBalancer Service
- Optional NGINX Ingress
- Kustomize configuration

## Prerequisites

Install and verify:

```bash
docker --version
kubectl version --client
minikube version
```

Docker must be running before Minikube is started with the Docker driver.

## Default image

The Deployment expects this local image:

```text
heart-disease-api:1.0.0
```

Build it from the repository root:

```bash
docker build -t heart-disease-api:1.0.0 .
```

## Start Minikube

```bash
minikube start --driver=docker --cpus=2 --memory=4096
```

Verify the cluster:

```bash
kubectl config current-context
kubectl get nodes
```

The current context should be `minikube`, and the node should be `Ready`.

## Load the local Docker image

```bash
minikube image load heart-disease-api:1.0.0
```

Verify that the image is present:

```bash
minikube image ls | grep heart-disease-api
```

On PowerShell:

```powershell
minikube image ls | Select-String "heart-disease-api"
```

## Deploy

From the repository root:

```bash
kubectl apply -k deployment/kubernetes
```

Wait for the rollout:

```bash
kubectl rollout status \
  deployment/heart-disease-api \
  -n heart-disease \
  --timeout=180s
```

Inspect the deployment:

```bash
kubectl get pods -n heart-disease
kubectl get services -n heart-disease
kubectl get deployment -n heart-disease
```

Both Pods should report:

```text
1/1 Running
```

## Access locally

Use port forwarding:

```bash
kubectl port-forward \
  service/heart-disease-api \
  8000:80 \
  -n heart-disease
```

PowerShell single-line version:

```powershell
kubectl port-forward service/heart-disease-api 8000:80 -n heart-disease
```

Available endpoints:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Liveness: `http://127.0.0.1:8000/health/live`
- Readiness: `http://127.0.0.1:8000/health/ready`
- Prediction: `http://127.0.0.1:8000/predict`
- Metrics: `http://127.0.0.1:8000/metrics`

## Test prediction

PowerShell:

```powershell
$body = Get-Content sample_request.json -Raw

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType application/json `
  -Body $body |
  ConvertTo-Json
```

Linux or macOS:

```bash
curl \
  --request POST \
  --header "Content-Type: application/json" \
  --data @sample_request.json \
  http://127.0.0.1:8000/predict
```

## LoadBalancer access

The Service is declared as `LoadBalancer`.

For Minikube, run this in a separate administrative terminal:

```bash
minikube tunnel
```

Then inspect:

```bash
kubectl get service heart-disease-api -n heart-disease
```

Port forwarding is usually simpler for local proof and screenshots.

## Optional Ingress

`ingress.yaml` requires an NGINX Ingress controller.

Enable the Minikube controller:

```bash
minikube addons enable ingress
```

Apply the Ingress:

```bash
kubectl apply -f deployment/kubernetes/ingress.yaml
```

Retrieve the Minikube IP:

```bash
minikube ip
```

Map it locally to:

```text
heart-disease.local
```

Then access:

```text
http://heart-disease.local/docs
```

## Use a public registry image

For a public deployment, update the image without editing the manifest:

```bash
kubectl set image \
  deployment/heart-disease-api \
  heart-disease-api=ghcr.io/2024ac05147-bits/heart-disease-api:1.0.0 \
  -n heart-disease
```

Then verify:

```bash
kubectl rollout status \
  deployment/heart-disease-api \
  -n heart-disease
```

## Remove the deployment

```bash
kubectl delete -k deployment/kubernetes
minikube stop
```