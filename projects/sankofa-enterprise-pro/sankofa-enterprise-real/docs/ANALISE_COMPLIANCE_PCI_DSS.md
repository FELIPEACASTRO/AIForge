# Análise de Compliance - PCI DSS 4.0

**Fonte:** [PCI DSS v4.0](https://www.pcisecuritystandards.org/documents/PCI-DSS-v4_0-PT.pdf)

## Objetivo do Padrão

O **Payment Card Industry Data Security Standard (PCI DSS)** é um padrão global que estabelece um conjunto de requisitos técnicos e operacionais para proteger os dados de contas de pagamento. Embora o Sankofa Enterprise Pro não armazene dados completos do cartão (como o PAN completo), ele processa dados de transações que estão no escopo do PCI DSS. Portanto, a conformidade é crucial.

## Os 12 Requisitos do PCI DSS e sua Aplicação ao Sankofa

O PCI DSS é organizado em 6 objetivos de controle, que se desdobram em 12 requisitos principais. A seguir, uma análise de como cada um se aplica ao Sankofa Enterprise Pro:

### Objetivo 1: Construir e Manter uma Rede e Sistemas Seguros

- **Requisito 1: Instalar e Manter Controles de Segurança de Rede:**
  - **Aplicação:** A infraestrutura AWS do Sankofa deve ser protegida por firewalls (Security Groups, NACLs) e uma arquitetura de rede segura (VPC, sub-redes privadas). O acesso à rede deve ser estritamente controlado.

- **Requisito 2: Aplicar as Configurações de Segurança para Todos os Componentes do Sistema:**
  - **Aplicação:** Todos os servidores, bancos de dados e outros componentes da infraestrutura devem ter configurações de segurança robustas (hardening). Senhas padrão devem ser alteradas e funcionalidades desnecessárias, desabilitadas.

### Objetivo 2: Proteger os Dados da Conta

- **Requisito 3: Proteger os Dados da Conta Armazenados:**
  - **Aplicação:** Embora o Sankofa não deva armazenar o PAN completo, qualquer dado de titular de cartão que seja armazenado (mesmo que temporariamente em logs ou cache) deve ser protegido. O PAN deve ser mascarado (exibindo no máximo os primeiros seis e os últimos quatro dígitos) e dados sensíveis de autenticação (como o CVV2) **nunca** devem ser armazenados.

- **Requisito 4: Proteger os Dados do Titular do Cartão com Criptografia Forte Durante a Transmissão:**
  - **Aplicação:** Toda a comunicação que envolve dados de transação, tanto interna (entre microsserviços) quanto externa (com o cliente), deve ser criptografada com protocolos fortes (TLS 1.2 ou superior). O uso de HTTPS é mandatório.

### Objetivo 3: Manter um Programa de Gestão de Vulnerabilidade

- **Requisito 5: Proteger Todos os Sistemas e Redes de Software Malicioso:**
  - **Aplicação:** Todos os sistemas devem ter software antivírus/antimalware instalado e atualizado. Processos para identificar e mitigar ameaças de malware devem estar em vigor.

- **Requisito 6: Desenvolver e Manter Sistemas e Software Seguros:**
  - **Aplicação:** O ciclo de vida de desenvolvimento de software (SDLC) do Sankofa deve ser seguro. Isso inclui a realização de revisões de código, a correção de vulnerabilidades (como as do OWASP Top 10) e a aplicação de patches de segurança em tempo hábil.

### Objetivo 4: Implementar Medidas Fortes de Controle de Acesso

- **Requisito 7: Restringir o Acesso aos Componentes do Sistema e aos Dados do Titular do Cartão por Necessidade de Conhecimento:**
  - **Aplicação:** O acesso aos dados de transação e aos sistemas do Sankofa deve ser baseado no princípio do menor privilégio. Apenas usuários autorizados com uma necessidade de negócio legítima devem ter acesso.

- **Requisito 8: Identificar Usuários e Autenticar o Acesso aos Componentes do Sistema:**
  - **Aplicação:** Cada usuário que acessa o sistema deve ter um ID único. A autenticação deve ser robusta, exigindo senhas fortes e, para acesso a ambientes críticos, autenticação multifator (MFA).

- **Requisito 9: Restringir o Acesso Físico aos Dados do Titular do Cartão:**
  - **Aplicação:** Embora a infraestrutura esteja na AWS, este requisito se aplica ao controle de acesso físico aos escritórios e a qualquer mídia física que possa conter dados de cartão. Para a AWS, a responsabilidade é compartilhada, mas a empresa deve garantir que o acesso ao console da AWS seja rigorosamente controlado.

### Objetivo 5: Monitorar e Testar as Redes Regularmente

- **Requisito 10: Registrar e Monitorar Todo o Acesso aos Componentes do Sistema e Dados do Titular do Cartão:**
  - **Aplicação:** O Sankofa deve gerar logs de auditoria detalhados para todas as ações realizadas no sistema, especialmente aquelas que envolvem acesso a dados de transação. Esses logs devem ser monitorados para detectar atividades suspeitas.

- **Requisito 11: Testar a Segurança de Sistemas e Redes Regularmente:**
  - **Aplicação:** A segurança do Sankofa deve ser testada regularmente, por meio de varreduras de vulnerabilidade, testes de penetração (pentests) e outras avaliações de segurança.

### Objetivo 6: Manter uma Política de Segurança da Informação

- **Requisito 12: Apoiar a Segurança da Informação com Políticas e Programas Organizacionais:**
  - **Aplicação:** A empresa deve ter uma política de segurança da informação abrangente que defina as responsabilidades de segurança para todos os funcionários e terceiros. Um programa de conscientização em segurança também é necessário.

## Implicações para o Sankofa Enterprise Pro

1.  **Escopo do CDE:** É fundamental definir claramente o **Ambiente de Dados do Titular do Cartão (CDE)** do Sankofa. Todos os sistemas que processam, armazenam ou transmitem dados de cartão fazem parte do CDE e devem estar em conformidade com o PCI DSS.
2.  **Tokenização:** Para reduzir o escopo do PCI DSS, o Sankofa deve considerar o uso de **tokenização**. Em vez de processar o PAN real, o sistema pode operar com tokens não sensíveis, o que simplifica significativamente a conformidade.
3.  **Segurança da Infraestrutura:** A segurança da infraestrutura AWS é a base da conformidade com o PCI DSS. A configuração correta de VPCs, Security Groups, IAM e outros serviços de segurança da AWS é essencial.
4.  **Desenvolvimento Seguro:** A equipe de desenvolvimento deve ser treinada em práticas de codificação segura para evitar a introdução de vulnerabilidades no software.
5.  **Auditoria e Monitoramento:** O sistema de monitoramento (DataDog) e as trilhas de auditoria devem ser configurados para atender aos requisitos de log e monitoramento do PCI DSS.
