# Flagger

## Description

Flagger is a Kubernetes progressive-delivery operator that automates the release process for applications running on Kubernetes. It reduces the risk of introducing a new software version into production by gradually shifting traffic to the new version while measuring metrics and running conformance tests. It is a graduated project of the Cloud Native Computing Foundation (CNCF) and part of the Flux family of GitOps tools. Its unique value proposition lies in its nature as a Kubernetes operator that integrates seamlessly with the Flux/GitOps ecosystem, focusing on automating metric analysis for promotion or rollback.

## Statistics

GitHub stars: 5.2k. CNCF status: Graduated Project (part of Flux). Deployment strategies: Canary, A/B Testing, Blue/Green.

## Features

Deployment strategies: Canary releases, A/B testing, Blue/Green mirroring. Release analysis: Queries Prometheus, InfluxDB, Datadog, New Relic, CloudWatch, Stackdriver, or Graphite. Traffic routing: Integration with service meshes (Istio, Linkerd, Kuma) and ingress controllers (NGINX, Contour, Knative, Gateway API). Alerts: Support for Slack, MS Teams, Discord, and Rocket. GitOps compatibility: Designed to be used in GitOps pipelines with tools such as Flux CD.

## Use Cases

Reducing the risk of software releases in production. Automating conformance tests and metric analysis during deployment. Implementing progressive-delivery strategies (Canary, A/B Testing, Blue/Green) in Kubernetes environments with a GitOps focus.

## Integration

Canary Analysis is configured through a Custom Resource Definition (CRD) where Flagger monitors a new version of the Deployment. It then creates a Canary resource that defines traffic routing and metric checks. For Prometheus, a PromQL query is used to define the success criterion, for example: 'sum(rate(http_requests_total{job=\"podinfo-canary\"}[1m])) / sum(rate(http_requests_total{job=\"podinfo-primary\"}[1m])) > 1.05' for error-rate comparison. Flagger acts as a control loop that performs the analysis and traffic routing.

## URL

https://flagger.app/