# Análise de Compliance - Resolução Conjunta n° 6 do BACEN

**Fonte:** [Resolução Conjunta n° 6 de 23/5/2023](https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo?tipo=Resolu%C3%A7%C3%A3o%20Conjunta&numero=6)

## Objetivo da Norma

Esta resolução estabelece os requisitos para o **compartilhamento de dados e informações sobre indícios de fraudes** entre instituições financeiras, de pagamento e outras autorizadas a funcionar pelo Banco Central do Brasil (BACEN). O objetivo é subsidiar e fortalecer os procedimentos e controles internos de prevenção a fraudes.

## Requisitos Essenciais para o Sankofa Enterprise Pro

O sistema Sankofa deve ser adaptado para atender aos seguintes requisitos mandatórios:

### 1. Sistema Eletrônico de Compartilhamento

O Sankofa deve implementar ou se integrar a um sistema eletrônico que permita:

- **Registro de Fraudes:** Registrar dados e informações sobre ocorrências ou tentativas de fraude identificadas.
- **Alteração e Exclusão:** Permitir a modificação e remoção dos registros quando necessário (e.g., em caso de erros ou inconsistências).
- **Consulta:** Disponibilizar os dados registrados para consulta pelas instituições participantes.

### 2. Conteúdo Mínimo dos Registros

Cada registro de fraude no sistema deve conter, no mínimo:

- **Identificação do Fraudador:** A identificação de quem, segundo os indícios, executou ou tentou executar a fraude.
- **Descrição dos Indícios:** Um detalhamento dos indícios que levaram à suspeita de fraude.
- **Identificação da Instituição:** Qual instituição foi responsável por registrar a informação.
- **Dados da Conta Destinatária:** Em casos de transferências ou pagamentos, a identificação da conta de destino e de seu titular.

### 3. Consentimento do Cliente

As instituições devem obter **consentimento prévio, geral e em destaque** de seus clientes para permitir o registro e compartilhamento de seus dados em caso de suspeita de fraude. Este consentimento deve:

- Ser específico para a finalidade de prevenção a fraudes.
- Constar em uma cláusula destacada no contrato ou em um instrumento jurídico válido.
- A documentação comprobatória deve estar disponível para o BACEN.

### 4. Princípios a Serem Observados

O compartilhamento de dados deve seguir os seguintes princípios:

- **Segurança e Privacidade:** Garantir a confidencialidade, integridade e disponibilidade dos dados.
- **Qualidade dos Dados:** Assegurar que as informações compartilhadas sejam precisas e relevantes.
- **Acesso Não Discriminatório:** Todas as instituições devem ter acesso pleno e igualitário ao sistema.
- **Eficiência e Padrão Único:** Adotar um padrão de comunicação comum para garantir a interoperabilidade.
- **Reciprocidade:** As instituições devem contribuir com dados para poderem consultar.
- **Interoperabilidade:** O sistema deve ser capaz de se comunicar com outros sistemas de compartilhamento de fraude existentes.

### 5. Requisitos Técnicos do Sistema

O sistema eletrônico deve:

- **Permitir Acesso Pleno:** Com identificação de quem realizou cada acesso.
- **Adotar Padrão Único de Comunicação:** Para garantir a interoperabilidade.
- **Assegurar Controles de Segurança:**
    - Confidencialidade, integridade e disponibilidade dos dados.
    - Aderência a certificações de segurança.
    - Realização de auditorias por empresas independentes.
    - Monitoramento das funcionalidades.
    - Segregação lógica ou física dos dados.
    - Controles de acesso robustos.
    - Garantir ao titular dos dados o livre acesso, correção e exclusão de suas informações.

## Implicações para o Sankofa Enterprise Pro

1.  **Módulo de Compliance:** É necessário criar um novo módulo no Sankofa dedicado à gestão de compliance com a Resolução Conjunta n° 6. Este módulo será responsável por:
    - Gerenciar a conexão com o sistema de compartilhamento de dados (ou implementar um, se necessário).
    - Formatar os dados de fraude do Sankofa para o padrão exigido.
    - Registrar, consultar, alterar e excluir informações de fraude.
    - Gerenciar o consentimento dos clientes.

2.  **Integração com a API:** A API do Sankofa deve ser estendida para expor endpoints que permitam a interação com este novo módulo de compliance, sempre com a devida autenticação e autorização.

3.  **Segurança e Auditoria:** O sistema de segurança e os logs de auditoria já implementados devem ser aprimorados para cobrir todas as interações relacionadas ao compartilhamento de dados de fraude, garantindo a rastreabilidade e a conformidade com a LGPD e com a própria resolução.
