# Data Transformation Prompts

## Description
Prompts de Transformação de Dados são instruções de Engenharia de Prompt projetadas para guiar Modelos de Linguagem Grande (LLMs) na conversão, limpeza, normalização e reestruturação de dados de um formato ou estado para outro. Essa técnica é fundamental para tarefas de pré-processamento de dados, onde a entrada (muitas vezes desestruturada, inconsistente ou em um formato específico) precisa ser convertida em uma saída estruturada e utilizável (como JSON, CSV, SQL, ou um formato de esquema específico). A eficácia reside em definir claramente o formato de entrada, o formato de saída desejado e as regras de manipulação ou limpeza a serem aplicadas. É amplamente utilizada em fluxos de trabalho de Engenharia de Dados e Análise de Dados para automatizar tarefas repetitivas e garantir a qualidade e a consistência dos dados.

## Examples
```
**1. Conversão de Formato (CSV para JSON):**
```
Aja como um conversor de dados. Sua tarefa é converter o texto CSV fornecido abaixo em um array de objetos JSON. Use os cabeçalhos do CSV como chaves do JSON.

CSV de Entrada:
Nome,Idade,Cidade
Alice,30,Nova York
Bob,25,Londres
Charlie,35,Paris
```

**2. Normalização de Dados (Padronização de Endereços):**
```
Você é um agente de limpeza de dados. Padronize a coluna 'Endereço' na lista fornecida para o formato 'Rua [Nome], Nº [Número], [Cidade], [Estado/País]'. Corrija abreviações e erros de digitação comuns.

Dados de Entrada:
- R. das Flores, 123, SP
- Av. Paulista 456, São Paulo
- 789 Oak St, NY, USA
```

**3. Extração e Reestruturação (Texto Não Estruturado para Tabela Markdown):**
```
Extraia as seguintes informações do texto abaixo: Nome do Produto, Preço e Disponibilidade. Apresente o resultado em uma tabela Markdown.

Texto de Entrada:
O novo Smartphone X, lançado em 2024, está disponível por R$ 4.500,00. O estoque é limitado. O Fone Y custa R$ 500 e está esgotado.
```

**4. Geração de Código SQL a partir de Requisitos:**
```
Com base no seguinte esquema de banco de dados (Tabela: Pedidos, Colunas: id_pedido, id_cliente, valor, data_pedido), escreva uma consulta SQL que retorne o 'id_cliente' e o 'valor' total de pedidos feitos no último mês.
```

**5. Limpeza de Texto (Remoção de Caracteres Especiais e Duplicatas):**
```
Limpe a lista de nomes de clientes a seguir. Remova quaisquer caracteres não alfanuméricos (exceto espaços) e elimine nomes duplicados. Retorne a lista limpa, um nome por linha.

Lista de Clientes:
João Silva!
Maria Souza
João Silva!
Pedro_Alves
Maria Souza
```

**6. Transformação de Unidades:**
```
Converta todos os valores de temperatura na lista a seguir de Celsius para Fahrenheit. Retorne apenas os novos valores.

Temperaturas em Celsius:
10, 25, 0, 37.5
```

**7. Filtragem e Agregação de Dados:**
```
Analise a lista de transações e filtre apenas as transações com 'status' = 'concluído'. Em seguida, calcule a soma total do 'valor' dessas transações.

Transações (Formato JSON):
[{"id": 1, "status": "pendente", "valor": 100}, {"id": 2, "status": "concluído", "valor": 250}, {"id": 3, "status": "concluído", "valor": 150}]
```
```

## Best Practices
**1. Definição Clara de Formato:** Sempre especifique o formato de saída desejado (e.g., "Retorne o resultado estritamente em formato JSON", "Converta para CSV com vírgulas como delimitador"). Use a notação de formato de saída (como JSON Schema) quando possível. **2. Fornecer Exemplos (Few-Shot):** Para transformações complexas ou ambíguas, inclua 1-2 exemplos de pares de entrada/saída para demonstrar o padrão de transformação esperado. **3. Instruções de Limpeza Explícitas:** Ao limpar dados, liste explicitamente as regras de limpeza (e.g., "Remova duplicatas", "Padronize datas para AAAA-MM-DD", "Substitua valores nulos por 'N/A'"). **4. Processamento em Lotes (Chunking):** Para grandes volumes de dados, divida a entrada em partes menores e use o prompt de transformação em cada parte, combinando os resultados posteriormente. Isso evita o estouro do limite de contexto do LLM. **5. Validação e Verificação:** Peça ao LLM para incluir uma etapa de validação ou um resumo das transformações realizadas, ou use ferramentas externas para validar o formato de saída (e.g., um validador JSON).

## Use Cases
**1. Engenharia de Dados (ETL/ELT):** Automatizar a conversão de dados brutos de logs ou sistemas legados (e.g., XML, texto plano) para formatos estruturados (e.g., JSON, Parquet) prontos para ingestão em data warehouses. **2. Limpeza e Pré-processamento de Dados:** Normalizar, padronizar e limpar conjuntos de dados para análise, corrigindo inconsistências, removendo duplicatas e tratando valores ausentes. **3. Geração de Código:** Criar scripts de transformação (Python, SQL, R) a partir de descrições em linguagem natural, acelerando o desenvolvimento de pipelines de dados. **4. Integração de Sistemas:** Converter formatos de mensagens entre diferentes APIs ou serviços (e.g., de um formato de resposta de API para um esquema de banco de dados interno). **5. Análise de Sentimento e Classificação:** Transformar texto não estruturado (e.g., avaliações de clientes) em dados categóricos ou numéricos (e.g., pontuação de sentimento, categoria do produto) para análise estatística.

## Pitfalls
**1. Ambiguidade de Formato:** Não especificar o formato de saída com clareza pode levar o LLM a retornar texto não estruturado ou um formato JSON/CSV inválido. **2. Limite de Contexto (Context Window):** Tentar transformar grandes conjuntos de dados de uma só vez pode exceder o limite de tokens do LLM, resultando em truncamento ou falha na transformação. **3. Erros de Tipagem de Dados:** O LLM pode interpretar incorretamente o tipo de dado (e.g., tratar um número como string) se as instruções de tipagem não forem explícitas. **4. Transformação Excessivamente Complexa:** Pedir ao LLM para realizar múltiplas transformações complexas (limpeza, conversão, agregação) em uma única etapa aumenta a chance de erros. É melhor usar o **Prompt Chaining** (encadeamento de prompts). **5. Alucinações em Limpeza:** Em vez de corrigir erros de dados, o LLM pode "alucinar" dados inexistentes ou fazer suposições incorretas se as regras de limpeza não forem estritas.

## URL
[https://stratpilot.ai/10-powerful-ai-prompts-for-data-transformation/](https://stratpilot.ai/10-powerful-ai-prompts-for-data-transformation/)
