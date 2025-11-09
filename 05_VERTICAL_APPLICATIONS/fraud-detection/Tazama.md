# Tazama

## Description

**Tazama** é a primeira plataforma de software de código aberto para monitoramento financeiro e detecção de fraudes em tempo real, gerenciada pela Linux Foundation Charities e financiada pela Fundação Gates. Sua proposta de valor única reside em oferecer uma solução **global, escalável e econômica** para o monitoramento de transações em tempo real, focando especialmente em **mercados emergentes** onde a fraude é prevalente devido à falta de ferramentas digitais seguras. O projeto visa promover a inclusão financeira, garantindo que os Provedores de Serviços Financeiros (FSPs) possam executar transações com segurança e rapidez, reduzindo o risco de fraudes e golpes. O nome "Tazama" vem do Swahili e significa "dar uma olhada", refletindo sua função de monitorar eventos em tempo real [1] [2]. O projeto foi lançado em fevereiro de 2024, com a versão 2.0 sendo a mais recente [3].

## Statistics

*   **Lançamento:** Fevereiro de 2024 (Versão 2.0) [3].
*   **Gerenciamento:** Linux Foundation Charities [1].
*   **Financiamento:** Fundação Gates [1].
*   **Foco:** Mercados emergentes, promovendo a inclusão financeira [2].
*   **Tecnologias Chave:** Docker, NATS, ArangoDB [6].
*   **Status do Repositório (GitHub - Full-Stack-Docker-Tazama):** 11 estrelas, 8 forks (em 08/11/2025) [6].
*   **Último Lançamento:** v2.2.0 (14 de agosto de 2025) [6].

## Features

*   **Monitoramento de Transações em Tempo Real:** Capacidade de monitorar cada transação à medida que ocorre, permitindo a detecção e prevenção imediata de fraudes.
*   **Prevenção de Fraudes e Golpes:** Software de última geração projetado para prevenir a "infecção" por fraudes e golpes, especialmente em sistemas de pagamento digital.
*   **Conformidade e AML:** Ajuda a aprimorar a conformidade regulatória e oferece recursos para detecção de lavagem de dinheiro (AML), conforme demonstrado em implementações bem-sucedidas [4].
*   **Arquitetura Baseada em Microsserviços:** Utiliza uma arquitetura desacoplada e baseada em microsserviços, o que a torna altamente escalável e adaptável a diferentes ambientes.
*   **Regras de Detecção Configuráveis:** Embora o código-fonte seja aberto, os "processadores de regras" privados (que contêm a lógica de detecção para evitar engenharia reversa por fraudadores) são implantados a partir do DockerHub com uma configuração genérica para acesso público, ou com acesso restrito para membros para uma configuração multi-tipologia completa [5].
*   **Tecnologias Modernas:** Construído com tecnologias como Docker, NATS (para mensagens em tempo real) e ArangoDB (para banco de dados) [6].

## Use Cases

*   **Monitoramento de Transações Financeiras Digitais:** O principal caso de uso é o monitoramento em tempo real de transações em sistemas de pagamento digital, como transferências e pagamentos móveis, para identificar e bloquear atividades fraudulentas antes que sejam concluídas [2].
*   **Prevenção de Lavagem de Dinheiro (AML):** O sistema é projetado para ajudar os Provedores de Serviços Financeiros (FSPs) a cumprir os requisitos de AML, detectando padrões de transação suspeitos que possam indicar lavagem de dinheiro [4].
*   **Apoio à Inclusão Financeira:** Ao fornecer uma solução de detecção de fraude de baixo custo e código aberto, o Tazama permite que FSPs em mercados emergentes ofereçam serviços financeiros digitais mais seguros, incentivando a adoção e a inclusão de populações não bancarizadas [2].
*   **Detecção de Fraudes Específicas:** O sistema é capaz de monitorar e reagir a vários tipos de fraudes e golpes, com a lógica de detecção contida em seus processadores de regras [5].

## Integration

A integração do Tazama é facilitada por sua arquitetura baseada em Docker e na comunicação via NATS (serviço de mensagens em tempo real). A implantação de pilha completa para demonstração e teste é feita usando `docker-compose`.

**Método de Integração (Exemplo de Configuração de Variáveis de Ambiente):**

A integração com o Serviço de Monitoramento de Transações (TMS) e outros componentes é configurada por meio de variáveis de ambiente, como visto no arquivo `ui.env` do repositório de demonstração [6]:

```javascript
// Exemplo de variáveis de ambiente para a interface de usuário (UI) de demonstração
NEXT_PUBLIC_URL="http://localhost:3001"
NEXT_PUBLIC_TMS_SERVER_URL="http://localhost:5000" // URL do serviço TMS
NEXT_PUBLIC_CMS_NATS_HOSTING="nats://nats:4222" // Conexão com o servidor NATS
NEXT_PUBLIC_ADMIN_SERVICE_HOSTING="http://localhost:5100" // URL do serviço Admin
NEXT_PUBLIC_ARANGO_DB_HOSTING="http://localhost:18529" // URL do banco de dados ArangoDB
NEXT_PUBLIC_EVENT_TYPES="['pacs.008.001.10', 'pacs.002.001.12', 'pain.001.001.11', 'pain.013.001.09']" // Tipos de eventos/transações monitorados
```

**Implantação com Docker Compose:**

Para iniciar uma instância de demonstração, o repositório `Full-Stack-Docker-Tazama` fornece scripts (`start.sh` para Unix ou `start.bat` para Windows) que utilizam arquivos `docker-compose` para orquestrar os microsserviços (TMS, Admin, NATS, ArangoDB, etc.) [6].

**Integração de Eventos:**

Os eventos de transação são enviados para o sistema Tazama, que os processa em tempo real usando o NATS para comunicação entre os microsserviços. Os tipos de eventos monitorados incluem padrões de mensagens financeiras como `pacs.008.001.10` e `pain.001.001.11` [6].

## URL

https://www.tazama.org/