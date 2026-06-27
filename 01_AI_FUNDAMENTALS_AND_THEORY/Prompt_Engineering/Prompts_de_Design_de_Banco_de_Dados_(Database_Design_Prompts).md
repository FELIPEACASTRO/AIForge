# Prompts de Design de Banco de Dados (Database Design Prompts)

## Description
**Prompts de Design de Banco de Dados** são técnicas de Engenharia de Prompt que utilizam Modelos de Linguagem Grande (LLMs) para auxiliar ou automatizar o processo de criação e otimização de esquemas de banco de dados. Essa categoria de prompts se concentra em fornecer à IA os requisitos de negócios, as entidades e os relacionamentos desejados, solicitando em troca a geração de código DDL (Data Definition Language), diagramas de relacionamento de entidades (ERD) ou recomendações de arquitetura.

A eficácia desses prompts reside na capacidade de a IA simular o raciocínio de um arquiteto de dados, aplicando princípios de normalização, estratégias de indexação e considerações de escalabilidade e segurança. Eles são particularmente úteis para acelerar a fase inicial de design, validar modelos conceituais e explorar diferentes abordagens de esquema (relacional, NoSQL, grafo) com base em casos de uso específicos. A tendência recente (2023-2025) mostra uma evolução de prompts simples para solicitações complexas que integram requisitos de conformidade (GDPR, LGPD) e arquiteturas de microsserviços.

## Examples
```
**Exemplo 1: Design de Esquema Relacional Completo (3NF)**
```
Atue como um Arquiteto de Banco de Dados Sênior. Projete um esquema de banco de dados relacional em 3ª Forma Normal (3NF) para uma plataforma de e-commerce que vende produtos digitais e físicos. O sistema deve gerenciar: Clientes, Pedidos, Itens do Pedido, Produtos, Categorias e Avaliações. Gere o código SQL DDL para PostgreSQL, incluindo chaves primárias, chaves estrangeiras e restrições NOT NULL.
```

**Exemplo 2: Otimização de Desempenho (Indexação)**
```
Dado o seguinte esquema de tabela [INSERIR CÓDIGO DDL DA TABELA AQUI], e sabendo que as consultas mais frequentes envolvem filtrar por 'status_pedido' e ordenar por 'data_criacao', sugira uma estratégia de indexação otimizada. Inclua a justificativa para o tipo de índice (B-tree, Hash, etc.) e o código SQL para criar os índices.
```

**Exemplo 3: Design NoSQL para Dados de Log**
```
Projete um modelo de dados NoSQL (MongoDB) para armazenar logs de acesso de um aplicativo web de alto tráfego. Cada log deve incluir: user_id, timestamp, endpoint_acessado, duracao_ms e dados_de_erro (se houver). O foco é na alta taxa de escrita e na recuperação rápida de logs por 'user_id' e 'timestamp'. Forneça o JSON de um documento de exemplo e a estrutura da coleção.
```

**Exemplo 4: Modelagem de Dados para Microsserviços**
```
Estamos migrando de um monolito para uma arquitetura de microsserviços. O microsserviço 'Inventário' é responsável por gerenciar o estoque de produtos. Projete o esquema de banco de dados (MySQL) para este microsserviço, garantindo que ele seja totalmente autônomo. O esquema deve suportar controle de estoque, localização de armazém e reservas de estoque. Gere o diagrama ER usando a sintaxe Mermaid.
```

**Exemplo 5: Inclusão de Requisitos de Conformidade (GDPR/LGPD)**
```
Projete a tabela 'Clientes' para um banco de dados de SaaS que opera na União Europeia e no Brasil. O design deve aderir aos princípios de Privacidade por Design (Privacy by Design) do GDPR/LGPD. Especifique quais campos devem ser criptografados (e.g., 'cpf', 'nome_completo'), como gerenciar o 'consentimento' e a estratégia para o 'direito ao esquecimento' (anonimização/exclusão).
```

**Exemplo 6: Refinamento de Relacionamento (Muitos-para-Muitos)**
```
Tenho as entidades 'Autores' e 'Livros' com um relacionamento de muitos-para-muitos. Crie a tabela de junção (join table) 'Autores_Livros' e adicione um campo extra chamado 'papel_autor' (e.g., 'Principal', 'Co-autor'). Gere o código DDL para esta tabela de junção em SQL Server.
```
```

## Best Practices
**1. Contextualização Detalhada:** Sempre forneça o máximo de detalhes sobre o projeto, incluindo o tipo de aplicação (e-commerce, SaaS, IoT), o volume de dados esperado (pequeno, médio, terabytes), e o foco principal (OLTP, OLAP, Híbrido).
**2. Especificação de Saída:** Peça explicitamente o formato de saída desejado (SQL DDL, Diagrama ER em Mermaid/PlantUML, JSON, Markdown).
**3. Restrições e Requisitos:** Inclua requisitos não funcionais cruciais, como nível de normalização (3NF, desnormalizado), requisitos de segurança (criptografia de campos sensíveis), e escalabilidade (sharding, replicação).
**4. Iteração e Refinamento:** Use prompts subsequentes para refinar o design inicial. Por exemplo, "Refine o esquema para a tabela 'Pedidos' adicionando um índice composto em 'status' e 'data_pedido'".
**5. Definição de Papel:** Comece o prompt definindo o papel da IA, como "Atue como um Arquiteto de Banco de Dados Sênior com 15 anos de experiência em sistemas distribuídos".

## Use Cases
**1. Prototipagem Rápida (MVP):** Gerar rapidamente o esquema inicial de um banco de dados para um Produto Mínimo Viável (MVP), permitindo que os desenvolvedores comecem a codificar imediatamente.
**2. Validação de Arquitetura:** Validar um modelo de dados conceitual existente, pedindo à IA para identificar falhas de normalização, gargalos de desempenho ou problemas de escalabilidade.
**3. Migração de SGBD:** Solicitar a conversão de um esquema de um SGBD para outro (e.g., de Oracle para PostgreSQL), incluindo a adaptação de tipos de dados e sintaxe DDL.
**4. Documentação Automática:** Gerar diagramas ERD (usando sintaxe como Mermaid ou PlantUML) e documentação detalhada do esquema a partir de uma descrição de alto nível.
**5. Otimização de Consultas:** Receber sugestões de índices, particionamento de tabelas e otimizações de esquema para melhorar o desempenho de consultas lentas em bases de dados existentes.
**6. Conformidade e Segurança:** Integrar requisitos de segurança e conformidade (e.g., PCI DSS, HIPAA, LGPD) diretamente no design do esquema, especificando campos para criptografia ou anonimização.

## Pitfalls
**1. Falta de Contexto:** Solicitar um design de banco de dados sem especificar o SGBD (PostgreSQL, MySQL, MongoDB), o volume de dados ou o tipo de carga de trabalho (OLTP vs. OLAP) leva a um design genérico e ineficiente.
**2. Confiança Excessiva na Normalização:** A IA pode sugerir um esquema altamente normalizado (4NF ou 5NF), o que é academicamente correto, mas pode introduzir complexidade e lentidão desnecessárias em sistemas de alto desempenho que se beneficiariam da desnormalização estratégica.
**3. Ignorar Requisitos Não Funcionais:** Não incluir requisitos de segurança, conformidade (GDPR/LGPD) ou estratégias de backup/recuperação no prompt inicial resulta em um design incompleto que exigirá retrabalho significativo.
**4. Ambiguidade de Entidades:** Usar nomes ambíguos ou não definir claramente as entidades e seus atributos (e.g., o que exatamente é um "Produto"?) fará com que a IA crie um modelo que não reflete a lógica de negócios real.
**5. Falha em Iterar:** Tratar o primeiro resultado da IA como o design final. O design de banco de dados é um processo iterativo; o prompt inicial deve ser seguido por prompts de refinamento e validação.

## URL
[https://clickup.com/p/ai-prompts/database-design](https://clickup.com/p/ai-prompts/database-design)
