# Prompt Engineering para Arquitetura de Nuvem (AWS, Azure, GCP)

## Description
A Engenharia de Prompt aplicada à Arquitetura de Nuvem é a prática de criar instruções otimizadas para Modelos de Linguagem Grande (LLMs) com o objetivo de gerar, analisar, otimizar e documentar soluções de infraestrutura em plataformas como AWS, Azure e Google Cloud. Essa técnica permite que arquitetos e engenheiros de nuvem acelerem o design de soluções, a criação de diagramas, a otimização de custos e a elaboração de documentação técnica, transformando requisitos de negócios em blueprints de nuvem precisos e eficientes. O foco está em fornecer contexto detalhado (serviços, requisitos de segurança, orçamento) para obter saídas estruturadas e acionáveis.

## Examples
```
1. **Geração de Diagrama de Arquitetura (AWS):** 'Atue como um Arquiteto de Soluções AWS Sênior. Gere um diagrama de arquitetura em formato Mermaid.js para uma aplicação web de alta disponibilidade e tolerância a falhas. A aplicação deve usar Amazon EC2 em um Auto Scaling Group distribuído por duas AZs, um Application Load Balancer, Amazon RDS Multi-AZ para o banco de dados e Amazon S3 para ativos estáticos. Inclua as sub-redes públicas e privadas e os Security Groups essenciais.'
2. **Otimização de Custos (GCP):** 'Analise a seguinte lista de recursos do Google Cloud e sugira 3 táticas de otimização de custos. O foco deve ser em instâncias de VM subutilizadas e opções de armazenamento mais baratas. [Lista de recursos: 2x n2-standard-4 rodando 24/7, 5TB de Standard Storage no Cloud Storage, 1x Cloud SQL com 99% de uptime]. Formate a resposta como uma tabela com 'Recurso', 'Tática Sugerida' e 'Economia Estimada (Mensal)'.'
3. **Revisão de Segurança (Azure):** 'Reescreva o seguinte trecho de política de segurança do Azure para torná-lo mais claro e conciso, garantindo que a conformidade com o CIS Benchmark seja mantida. O trecho é: [Trecho da política]. Além disso, identifique um serviço do Azure (como Azure Policy ou Azure Security Center) que possa automatizar a aplicação desta regra.'
4. **Comparação Multi-Cloud:** 'Compare os serviços de contêiner gerenciado (Amazon EKS, Azure AKS, Google GKE) em termos de complexidade de gerenciamento, modelo de precificação e integração com ferramentas de CI/CD. O público-alvo é uma startup com foco em DevOps. Apresente a comparação em formato de lista numerada, com uma recomendação final justificada.'
5. **Resolução de Problemas:** 'Recebemos um alerta de alta latência no nosso Application Gateway (Azure). Liste 5 possíveis causas e, para cada uma, forneça um comando ou ação de diagnóstico inicial no Azure CLI.'
6. **Criação de Documentação Técnica:** 'Com base no seguinte arquivo Terraform (fornecido no contexto), gere a seção 'Visão Geral da Arquitetura' para o documento de design da solução. O público-alvo são engenheiros de nível júnior. Use linguagem simples e inclua uma breve explicação de cada recurso principal.'
```

## Best Practices
1. **Definir o Papel (Persona):** Comece o prompt instruindo o LLM a atuar como um 'Arquiteto de Soluções Sênior', 'Especialista em FinOps' ou 'Engenheiro de Segurança de Nuvem'.
2. **Especificar a Plataforma e o Serviço:** Seja explícito sobre a plataforma (AWS, Azure, GCP) e os serviços específicos (ex: 'AWS Lambda', 'Azure Cosmos DB', 'GCP Cloud Run').
3. **Usar Formato Estruturado para Saída:** Solicite a saída em formatos fáceis de analisar, como tabelas, JSON, YAML, ou linguagens de diagrama como Mermaid.js ou PlantUML.
4. **Fornecer Contexto de Negócio:** Inclua requisitos não funcionais (escalabilidade, custo, segurança, latência) e o caso de uso (e-commerce, IoT, data pipeline) para guiar a solução.
5. **Iteração e Refinamento:** Use a saída inicial como base para prompts de refinamento (ex: 'Refine esta arquitetura para reduzir o custo em 20%', 'Adicione um WAF à frente do Load Balancer').

## Use Cases
1. **Geração Rápida de Blueprints:** Criar rascunhos de arquitetura para novas aplicações ou migrações.
2. **Otimização de Custos (FinOps):** Identificar e sugerir alterações em recursos de nuvem para reduzir despesas.
3. **Documentação Automatizada:** Gerar documentação técnica (visões gerais, guias de implantação) a partir de código de Infraestrutura como Código (IaC) ou descrições de alto nível.
4. **Análise de Conformidade e Segurança:** Avaliar uma arquitetura proposta em relação a padrões de segurança (CIS, NIST) ou regulamentações (LGPD, HIPAA).
5. **Planejamento de Migração Multi-Cloud:** Comparar serviços e gerar estratégias de migração entre diferentes provedores de nuvem.
6. **Geração de Código IaC:** Criar snippets de código Terraform, CloudFormation ou ARM Templates para recursos específicos.

## Pitfalls
1. **Ambiguidade de Serviço:** Usar nomes de serviços genéricos (ex: 'banco de dados') sem especificar o tipo (ex: 'Amazon RDS Aurora Serverless') leva a resultados imprecisos.
2. **Ignorar o Contexto de Segurança:** Não incluir requisitos de segurança ou conformidade no prompt pode resultar em arquiteturas funcionais, mas inseguras.
3. **Confiança Excessiva:** Aceitar a saída do LLM sem validação manual. A IA pode 'alucinar' (inventar) serviços, configurações ou comandos inexistentes.
4. **Falta de Formato de Saída:** Não especificar um formato de saída (ex: 'Liste os passos') resulta em texto longo e difícil de parsear ou usar em automação.
5. **Prompts Longos e Complexos:** Tentar incluir muitos requisitos em um único prompt pode confundir o modelo. É melhor usar uma abordagem de múltiplos passos (Chain-of-Thought).

## URL
[https://medium.com/@dave-patten/prompt-engineering-for-architects-making-ai-speak-architecture-d812648cf755](https://medium.com/@dave-patten/prompt-engineering-for-architects-making-ai-speak-architecture-d812648cf755)
