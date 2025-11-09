# Flagger

## Description

Flagger é um operador Kubernetes de entrega progressiva que automatiza o processo de lançamento para aplicações em execução no Kubernetes. Reduz o risco de introduzir uma nova versão de software em produção, mudando gradualmente o tráfego para a nova versão enquanto mede métricas e executa testes de conformidade. É um projeto graduado da Cloud Native Computing Foundation (CNCF) e parte da família de ferramentas GitOps do Flux. Sua proposta de valor única reside na sua natureza de operador Kubernetes que se integra perfeitamente com o ecossistema Flux/GitOps, focando na automação da análise de métricas para a promoção ou rollback.

## Statistics

Estrelas no GitHub: 5.2k. Status CNCF: Projeto Graduado (parte do Flux). Estratégias de Implantação: Canary, A/B Testing, Blue/Green.

## Features

Estratégias de implantação: Canary releases, A/B testing, Blue/Green mirroring. Análise de lançamento: Consulta Prometheus, InfluxDB, Datadog, New Relic, CloudWatch, Stackdriver ou Graphite. Roteamento de tráfego: Integração com service meshes (Istio, Linkerd, Kuma) e ingress controllers (NGINX, Contour, Knative, Gateway API). Alertas: Suporte para Slack, MS Teams, Discord e Rocket. Compatibilidade GitOps: Projetado para ser usado em pipelines GitOps com ferramentas como Flux CD.

## Use Cases

Redução do risco de lançamentos de software em produção. Automação de testes de conformidade e análise de métricas durante a implantação. Implementação de estratégias de entrega progressiva (Canary, A/B Testing, Blue/Green) em ambientes Kubernetes com foco em GitOps.

## Integration

A Análise Canary é configurada através de um Custom Resource Definition (CRD) onde o Flagger monitora uma nova versão do Deployment. Ele então cria um recurso Canary que define o roteamento de tráfego e as verificações de métricas. Para o Prometheus, uma consulta PromQL é usada para definir o critério de sucesso, por exemplo: 'sum(rate(http_requests_total{job=\"podinfo-canary\"}[1m])) / sum(rate(http_requests_total{job=\"podinfo-primary\"}[1m])) > 1.05' para comparação da taxa de erro. O Flagger atua como um loop de controle que executa a análise e o roteamento de tráfego.

## URL

https://flagger.app/