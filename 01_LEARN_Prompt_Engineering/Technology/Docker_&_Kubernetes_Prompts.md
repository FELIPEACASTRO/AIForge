# Docker & Kubernetes Prompts

## Description
**Prompts de Docker & Kubernetes** é uma categoria de Prompt Engineering focada na utilização de Modelos de Linguagem Grande (LLMs) para automatizar, otimizar e solucionar problemas em ambientes de desenvolvimento e produção baseados em contêineres e orquestração. Esta técnica se insere no contexto de **PromptOps** ou **DevOps assistido por IA**, onde a IA atua como um copiloto para engenheiros de DevOps, SREs e desenvolvedores. O objetivo principal é acelerar a criação de arquivos de configuração (como `Dockerfile` e manifestos YAML de Kubernetes), diagnosticar falhas complexas e gerar scripts de automação, transformando descrições em linguagem natural em código de infraestrutura acionável. A eficácia desta técnica depende da clareza, especificidade e do fornecimento de contexto técnico detalhado nos prompts.

## Examples
```
1.  **Geração de Dockerfile Otimizado (Multi-Stage):**
    ```
    Crie um Dockerfile multi-stage para uma aplicação Python (Flask) que usa a imagem base 'python:3.11-slim'. O estágio de build deve instalar as dependências de 'requirements.txt'. O estágio final deve usar 'python:3.11-slim' e copiar apenas o código-fonte e as dependências instaladas. Garanta que o usuário de execução final seja não-root e que o cache do pip seja limpo.
    ```

2.  **Geração de Manifesto Kubernetes (Deployment e Service):**
    ```
    Gere um manifesto YAML de Kubernetes que inclua um Deployment e um Service. O Deployment deve ter 3 réplicas, usar a imagem 'minha-app:v1.2.0' e expor a porta 8080. O Service deve ser do tipo LoadBalancer e rotear o tráfego para o Deployment. Adicione um readinessProbe que verifique o endpoint '/health' na porta 8080.
    ```

3.  **Solução de Problemas (CrashLoopBackOff):**
    ```
    Estou recebendo o erro 'CrashLoopBackOff' no meu Pod. Analise os logs do container (logs anexados abaixo) e o manifesto YAML do Deployment (também anexado). Identifique a causa provável e sugira a correção exata no manifesto YAML.

    [Logs do Container]
    ...
    [Manifesto YAML]
    ...
    ```

4.  **Otimização de Dockerfile Existente:**
    ```
    Analise o Dockerfile fornecido abaixo. Sugira 3 otimizações para reduzir o tamanho final da imagem e o tempo de build, focando em cache de camadas e melhores práticas de segurança. Apresente o Dockerfile otimizado.

    [Dockerfile Existente]
    ...
    ```

5.  **Criação de Configuração de Ingress:**
    ```
    Crie um manifesto Ingress para Kubernetes que roteie o tráfego do host 'api.meudominio.com' para o Service chamado 'api-service' na porta 80. O Ingress deve usar TLS com um Secret chamado 'meu-tls-secret'.
    ```

6.  **Geração de Script de Shell para K8s:**
    ```
    Escreva um script de shell que verifique o status de todos os Pods no namespace 'producao'. Se algum Pod estiver em estado 'CrashLoopBackOff' ou 'ImagePullBackOff', o script deve imprimir o nome do Pod e seus logs recentes.
    ```

7.  **Explicação e Geração de HPA:**
    ```
    Explique o conceito de Horizontal Pod Autoscaler (HPA) no Kubernetes. Em seguida, gere um manifesto HPA para o Deployment chamado 'web-app-deployment' que mantenha o uso médio de CPU em 70%, com um mínimo de 2 e um máximo de 10 réplicas.
    ```

8.  **Validação e Correção de YAML:**
    ```
    Valide o manifesto YAML de Kubernetes abaixo. Corrija quaisquer erros de sintaxe, indentação ou versão de API. Mantenha a lógica original intacta e retorne apenas o YAML corrigido.

    [YAML com Erro]
    ...
    ```

9.  **Geração de ConfigMap para Variáveis de Ambiente:**
    ```
    Crie um ConfigMap chamado 'app-config' com as seguintes variáveis de ambiente: 'LOG_LEVEL'='INFO', 'FEATURE_TOGGLE'='true', e 'API_URL'='http://backend-service'. Em seguida, mostre como referenciar este ConfigMap em um Deployment YAML.
    ```

10. **Revisão de Segurança de Dockerfile:**
    ```
    Revise o Dockerfile abaixo para identificar e corrigir vulnerabilidades de segurança. As correções devem incluir a remoção de senhas ou chaves expostas, a garantia de que o usuário não-root seja usado e a atualização de pacotes desatualizados.

    [Dockerfile para Revisão]
    ...
    ```
```

## Best Practices
**Clareza e Especificidade Extrema:** Trate o LLM como um "engenheiro júnior brilhante, mas não confiável". Seja extremamente detalhado sobre a versão da linguagem, a imagem base, os requisitos de segurança (ex: usuário não-root), e o tipo exato de recurso (ex: Deployment, Service, Ingress).
**Comunicação Estruturada e Iterativa:** Use prompts iterativos. Primeiro, peça a geração do código. Em seguida, peça a validação e a correção de erros. Por fim, peça a otimização (ex: "Agora, otimize este Dockerfile para uma imagem final menor").
**Inclusão de Contexto e Restrições:** Sempre forneça o contexto relevante (logs de erro, código existente, requisitos de rede) e restrições (ex: "O Service deve ser do tipo LoadBalancer", "O HPA deve manter o uso de CPU em 70%").
**Validação e Segurança:** Peça explicitamente ao LLM para validar o código gerado (ex: "Verifique a sintaxe YAML e a versão da API") e para aplicar as melhores práticas de segurança (ex: "Adicione um healthcheck e garanta que o usuário não-root seja usado").

## Use Cases
**Automação de Configuração:** Geração rápida de `Dockerfile` otimizados (multi-stage, base image leve) e manifestos YAML de Kubernetes (Deployment, Service, Ingress, HPA) a partir de requisitos em linguagem natural.
**Solução de Problemas (Troubleshooting):** Diagnóstico e sugestão de correção para erros comuns de Kubernetes (ex: `CrashLoopBackOff`, `ImagePullBackOff`) com base em logs e manifestos fornecidos.
**Otimização de Infraestrutura:** Otimização de Dockerfiles para redução do tamanho da imagem e do tempo de build, e sugestão de configurações de Kubernetes para melhor escalabilidade e resiliência.
**Geração de Scripts e Documentação:** Criação de scripts de shell para tarefas de manutenção e automação de K8s, e geração de documentação técnica a partir de configurações existentes.
**Revisão de Segurança:** Análise de Dockerfiles e manifestos de Kubernetes para identificar e corrigir vulnerabilidades de segurança (ex: uso de usuário root, exposição de segredos).

## Pitfalls
**Confiança Cega no Código Gerado:** O LLM pode gerar YAMLs com versões de API desatualizadas, Dockerfiles sem `healthcheck` ou configurações inseguras. A validação manual ou por ferramentas de IA específicas (como K8sGPT) é crucial.
**Exposição de Dados Sensíveis:** Inserir logs de produção, segredos ou dados proprietários em prompts de modelos de IA públicos pode violar políticas de segurança e contratos de confidencialidade. Recomenda-se o uso de modelos hospedados localmente ou com garantias de privacidade.
**Prompts Vagos:** Prompts genéricos resultam em código não otimizado, inseguro ou que não atende aos requisitos específicos do ambiente de produção. A falta de contexto (ex: imagem base, versão da linguagem) leva a resultados de baixa qualidade.
**Ignorar a Iteração:** Esperar o resultado perfeito no primeiro prompt. A engenharia de prompts para DevOps é um processo iterativo de refinamento e correção.

## URL
[https://medium.com/@osomudeyazudonu/10-ai-prompts-every-devops-engineer-should-use-to-work-10-faster-3474ac59ffc1](https://medium.com/@osomudeyazudonu/10-ai-prompts-every-devops-engineer-should-use-to-work-10-faster-3474ac59ffc1)
