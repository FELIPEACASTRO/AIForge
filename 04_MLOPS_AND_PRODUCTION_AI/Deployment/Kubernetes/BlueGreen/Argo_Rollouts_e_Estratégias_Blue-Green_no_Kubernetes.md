# Argo Rollouts e Estratégias Blue-Green no Kubernetes

## Description

Argo Rollouts é um controlador Kubernetes que estende as capacidades nativas de implantação, oferecendo estratégias avançadas como Blue-Green, Canary e A/B testing. A estratégia Blue-Green é um padrão de lançamento que visa reduzir o tempo de inatividade e o risco, mantendo duas versões idênticas do ambiente de produção (Blue - estável e Green - nova) e alternando o tráfego entre elas. O Argo Rollouts automatiza essa alternância e o processo de análise de risco, permitindo rollbacks instantâneos e zero-downtime. Ferramentas como Flagger também oferecem essa funcionalidade, muitas vezes integrando-se a Service Meshes como Istio ou Linkerd para controle de tráfego mais granular.

## Statistics

A adoção de estratégias de entrega progressiva como Blue-Green, facilitada por ferramentas como Argo Rollouts, está diretamente ligada a métricas de DevOps de alto desempenho, como: **Taxa de Falha de Mudança (Change Failure Rate)**, que é significativamente reduzida devido ao rollback instantâneo; **Tempo Médio de Recuperação (MTTR)**, que é minimizado pela capacidade de alternar o tráfego de volta para o ambiente 'Blue' estável; e **Frequência de Implantação**, que pode ser aumentada com segurança. O Argo Rollouts é um projeto de código aberto ativo, parte do ecossistema Argo, com milhares de estrelas no GitHub e uma grande comunidade de contribuidores, indicando alta confiança e maturidade.

## Features

1. **Estratégia Blue-Green Automatizada:** Gerencia a criação e a transição entre os ReplicaSets 'Blue' (estável) e 'Green' (nova versão). 2. **Rollback Instantâneo:** Capacidade de reverter o tráfego para a versão 'Blue' estável em caso de falha na nova versão. 3. **Análise de Rollout (Analysis):** Integração com ferramentas de monitoramento (Prometheus, Datadog, New Relic) para verificar métricas de saúde e desempenho antes da promoção final. 4. **Controle de Tráfego:** Suporte nativo para Ingress Controllers (como NGINX, ALB) e Service Meshes (Istio, Linkerd) para roteamento de tráfego. 5. **Webhooks:** Permite a integração com sistemas de CI/CD para notificação e controle do ciclo de vida da implantação.

## Use Cases

1. **Implantações de Missão Crítica:** Ideal para aplicações que exigem zero-downtime, como serviços financeiros ou e-commerce de alto tráfego. 2. **Testes de Aceitação do Usuário (UAT) em Produção:** Permite que um pequeno grupo de usuários ou testadores acesse a versão 'Green' antes da promoção total. 3. **Atualizações de Banco de Dados:** Embora a estratégia Blue-Green não resolva diretamente a migração de esquema de banco de dados, ela permite que a nova versão da aplicação seja testada com uma cópia do banco de dados de produção antes da alternância final. 4. **Redução de Risco:** Minimiza o risco de implantações, pois a versão anterior está sempre disponível para um rollback imediato.

## Integration

A integração é feita através da definição de um recurso `Rollout` no Kubernetes, que substitui o recurso `Deployment` padrão. O Rollout define a estratégia Blue-Green e as regras de análise. \n\n**Exemplo de Rollout Blue-Green (YAML):**\n```yaml\napiVersion: argoproj.io/v1alpha1\nkind: Rollout\nmetadata:\n  name: my-app-rollout\nspec:\n  replicas: 5\n  strategy:\n    blueGreen:\n      activeService: my-app-active\n      previewService: my-app-preview\n      autoPromotionEnabled: false # Requer promoção manual após o teste\n  selector:\n    matchLabels:\n      app: my-app\n  template:\n    metadata:\n      labels:\n        app: my-app\n    spec:\n      containers:\n      - name: my-app\n        image: my-app:v2.0.0 # Nova versão\n```\n\n**Passos de Integração:** 1. Instalar o Argo Rollouts Controller no cluster Kubernetes. 2. Criar dois serviços (ex: `my-app-active` e `my-app-preview`). 3. Definir o recurso `Rollout` acima. 4. O Rollout gerencia a alternância do seletor do serviço `activeService` para apontar para o novo ReplicaSet após a aprovação.

## URL

https://argoproj.io/projects/argo-rollouts/