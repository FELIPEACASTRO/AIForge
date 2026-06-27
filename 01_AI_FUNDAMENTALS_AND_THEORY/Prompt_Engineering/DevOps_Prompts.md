# DevOps Prompts

## Description
Prompt Engineering para DevOps é a prática de projetar e refinar prompts de forma estratégica para maximizar a utilidade de modelos de Linguagem Grande (LLMs) em tarefas de desenvolvimento, implantação e operações. Envolve a criação de instruções claras, contextuais e estruturadas para automatizar tarefas repetitivas, otimizar pipelines de CI/CD, gerar código e scripts de infraestrutura (IaC), depurar logs complexos e fortalecer a segurança. A aplicação correta do Prompt Engineering em DevOps visa aumentar a produtividade, reduzir o tempo de inatividade e garantir a entrega contínua de software de alta qualidade e escalável. É uma competência crucial para engenheiros DevOps que buscam integrar a Inteligência Artificial em seus fluxos de trabalho diários.

## Examples
```
**1. Geração de Script de Monitoramento (Shell):**
"Atue como um Engenheiro de Sistemas Linux. Crie um script Shell que monitore o uso de CPU, memória e disco (`top`, `df`, `free`) em um servidor Ubuntu 22.04. O script deve compilar as métricas em um formato de relatório simples e enviá-lo por e-mail para `alerta@empresa.com` se o uso de CPU exceder 80%."

**2. Criação de IaC (Terraform):**
"Gere um script Terraform para a AWS. O script deve provisionar um grupo de auto-escalonamento para servidores web, um Application Load Balancer (ALB) e um Security Group que permita apenas tráfego HTTP/HTTPS. O escalonamento deve ser baseado no uso de CPU e o código deve ser modular."

**3. Otimização de Pipeline CI/CD (GitLab CI):**
"Analise o seguinte arquivo `.gitlab-ci.yml` (fornecido no prompt). Sugira otimizações para reduzir o tempo de construção em 30%, focando na paralelização de testes e no cache de dependências. Apresente as sugestões como um novo arquivo YAML completo."

**4. Debugging de Logs (Kubernetes):**
"Desenvolva um prompt para um LLM que analise os logs de erro do Kubernetes (fornecidos no prompt) de um pod que está falhando ao iniciar. O prompt deve solicitar a causa raiz mais provável e uma solução passo a passo para mitigar o erro, formatando a saída em JSON."

**5. Análise de Segurança (Nginx):**
"Atue como um Engenheiro de Segurança DevOps. Analise o seguinte arquivo de configuração do Nginx (fornecido no prompt). Sugira melhorias de segurança para mitigar ataques OWASP Top 10, como `clickjacking` e `XSS`, formatando a saída como um checklist de ações a serem tomadas."

**6. Automação de Tarefas Manuais (Python):**
"Gere um script Python usando a biblioteca `boto3` para automatizar a rotação de chaves de acesso de um usuário IAM na AWS. O script deve criar uma nova chave, atualizar a chave em um sistema de gerenciamento de segredos (ex: AWS Secrets Manager) e revogar a chave antiga após 24 horas."

**7. Geração de Testes de Unidade (Jest):**
"Crie 5 casos de teste de unidade usando Jest para a seguinte função JavaScript (fornecida no prompt) que valida endereços de e-mail. Os testes devem cobrir casos de sucesso, falha, e-mails vazios e formatos inválidos."
```

## Best Practices
**1. Definição de Papel e Contexto:** Sempre comece o prompt definindo o papel da IA (ex: "Atue como um Engenheiro de Segurança DevOps") e forneça o máximo de contexto possível sobre o ambiente, tecnologia e objetivo.
**2. Estrutura de Saída Explícita:** Especifique o formato de saída desejado (ex: "Gere o código em um bloco Markdown YAML", "Responda em formato JSON com os campos 'causa_raiz' e 'solução'").
**3. Iteração e Refinamento:** Comece com prompts simples e adicione complexidade gradualmente. Use a saída anterior da IA como entrada para o próximo prompt para refinar o resultado.
**4. Validação Rigorosa:** Nunca implemente código, scripts ou configurações geradas pela IA em ambientes de produção sem uma revisão e validação humana completa.
**5. Inclusão de Restrições de Segurança:** Peça explicitamente à IA para seguir as melhores práticas de segurança (ex: "Garanta que o script não contenha credenciais em texto simples e siga o princípio do menor privilégio").

## Use Cases
**1. Otimização de Pipeline CI/CD:** Sugerir melhorias em arquivos YAML de pipeline (ex: Jenkins, GitLab CI, GitHub Actions) para reduzir o tempo de construção e aumentar a eficiência.
**2. Geração de Infraestrutura como Código (IaC):** Criar ou modificar templates Terraform, CloudFormation ou Ansible Playbooks para provisionamento e gerenciamento de infraestrutura.
**3. Debugging e Análise de Logs:** Analisar logs de erro complexos (ex: Kubernetes, logs de aplicação) para identificar a causa raiz de falhas e sugerir correções.
**4. Geração de Código e Scripts:** Criar trechos de código, scripts Shell, Python ou PowerShell para automação de tarefas operacionais e rotinas de manutenção.
**5. Segurança e Conformidade:** Identificar vulnerabilidades em configurações (ex: Nginx, Dockerfile) e gerar políticas de segurança ou scripts de auditoria.
**6. Documentação Técnica:** Gerar documentação detalhada a partir de código-fonte, logs de implantação ou diagramas de arquitetura.
**7. Monitoramento e Alerta:** Criar consultas e regras de alerta para ferramentas de monitoramento (ex: Prometheus, Grafana) com base em padrões de log ou métricas.
**8. Otimização de Custos em Nuvem:** Analisar relatórios de uso de recursos em nuvem e sugerir otimizações para reduzir custos.
**9. Resposta a Incidentes:** Analisar a linha do tempo de um incidente e sugerir etapas de mitigação e planos de ação pós-mortem.
**10. Geração de Casos de Teste:** Criar casos de teste de unidade, integração ou carga para garantir a qualidade do software.

## Pitfalls
**1. Prompt Injection e Vazamento de Dados:** Expor logs, configurações ou segredos sensíveis no prompt para depuração, o que pode levar a vazamento de dados. Além disso, a vulnerabilidade a ataques de *Prompt Injection* pode levar à execução de código não autorizado.
**2. Confiança Excessiva (Prompt and Pray):** Implementar cegamente a saída da IA (código, scripts, configurações) sem validação ou revisão humana, o que é crítico em ambientes de produção.
**3. Vaguedade e Ambiguidade:** Prompts mal definidos que levam a saídas inconsistentes, irrelevantes ou incorretas, exigindo retrabalho.
**4. Alucinações:** A IA pode gerar código ou informações factualmente incorretas que parecem plausíveis, mas não funcionam no ambiente real, causando falhas na implantação.
**5. Sobrecarga de Tarefas:** Tentar resolver muitos problemas ou solicitar muitas tarefas em um único prompt, o que diminui a precisão e a qualidade da resposta da IA.
**6. Código Inseguro:** A IA pode gerar código com vulnerabilidades de segurança se não for explicitamente instruída a seguir as melhores práticas de segurança (ex: permissões excessivas, senhas em texto simples).

## URL
[https://marutitech.com/what-is-prompt-engineering-devops/](https://marutitech.com/what-is-prompt-engineering-devops/)
