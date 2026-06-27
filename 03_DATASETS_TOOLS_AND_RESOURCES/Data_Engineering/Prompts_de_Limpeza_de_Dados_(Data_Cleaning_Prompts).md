# Prompts de Limpeza de Dados (Data Cleaning Prompts)

## Description
A técnica de **Prompts de Limpeza de Dados** (Data Cleaning Prompts) utiliza comandos em linguagem natural para instruir Grandes Modelos de Linguagem (LLMs) a identificar, corrigir, padronizar e remover erros ou inconsistências em conjuntos de dados. Essa abordagem transforma tarefas manuais e repetitivas, como preenchimento de valores ausentes, remoção de duplicatas e padronização de formatos, em processos automatizados e eficientes. Ao invés de escrever scripts complexos ou usar fórmulas extensas, o analista de dados ou engenheiro de ML pode simplesmente descrever a ação desejada (ex: "Preencha valores ausentes na Coluna D com a mediana da coluna"), permitindo que o LLM execute a lógica de limpeza com base em seu vasto conhecimento de padrões de dados e regras de formatação [1]. Isso acelera significativamente a fase crítica de preparação de dados, que historicamente consome a maior parte do tempo em projetos de análise e Machine Learning [1].

## Examples
```
1. **Preenchimento de Dados Ausentes**: "Preencha todos os valores ausentes na Coluna 'Idade' usando a mediana da coluna e justifique a escolha do método."
2. **Remoção de Duplicatas**: "Encontre e remova linhas duplicadas no conjunto de dados, considerando apenas as colunas 'ID do Cliente' e 'Data da Transação', e mantenha o registro mais recente."
3. **Padronização de Formato de Data**: "Converta todas as datas na Coluna 'Data de Início' para o formato ISO 8601 (YYYY-MM-DD)."
4. **Padronização de Texto (Case)**: "Padronize todos os nomes de produtos na Coluna 'Nome do Produto' para o formato Title Case (Primeira Letra de Cada Palavra em Maiúscula)."
5. **Correção de Erros de Digitação**: "Identifique e corrija erros de ortografia comuns e variações de nomes de cidades na Coluna 'Localização', usando uma lista de cidades brasileiras como referência."
6. **Detecção e Flag de Outliers**: "Sinalize todas as transações na Coluna 'Valor da Venda' que estejam 3 desvios-padrão acima ou abaixo da média, e liste as 5 maiores anomalias."
7. **Extração e Categorização**: "Extraia o código de área (DDD) dos números de telefone na Coluna 'Telefone' e crie uma nova coluna chamada 'DDD' com essa informação."
8. **Limpeza Condicional**: "Remova registros onde a Coluna 'Status' é 'Cancelado' E a Coluna 'Data de Cancelamento' está vazia."
```

## Best Practices
**Seja Claro e Específico**: Em vez de "Limpe meus dados", use "Encontre e remova linhas duplicadas na Coluna C, mantendo a primeira ocorrência" [1]. **Use Palavras de Ação**: Prefira verbos diretos como *Converter*, *Limpar*, *Encontrar*, *Remover*, *Padronizar* ou *Categorizar* [1]. **Especifique Colunas e Formatos**: Sempre defina o escopo da ação (ex: "na Coluna D") e o formato de saída desejado (ex: "para o formato YYYY-MM-DD") [1]. **Defina Condições**: Inclua lógica condicional quando necessário (ex: "Remova duplicatas apenas se Coluna B e Coluna C forem idênticas") [1]. **Processamento em Lotes (Chunking)**: Para grandes volumes de dados, divida o conjunto em partes menores (chunks) e processe-as sequencialmente ou em paralelo, pois LLMs têm limites de contexto [2]. **Validação Pós-Limpeza**: Use prompts para validar o resultado, como "Verifique se todos os emails na Coluna F contêm '@' e um domínio válido" [2].

## Use Cases
**Análise de Dados e Business Intelligence (BI)**: Assegurar que os dados de entrada para relatórios e painéis de BI sejam precisos e consistentes, evitando decisões baseadas em informações falhas. **Machine Learning (ML)**: Preparar e pré-processar rapidamente grandes volumes de dados para treinamento de modelos, padronizando *features* e tratando valores ausentes, o que é vital para a performance do modelo. **Migração de Sistemas e Integração de Dados**: Padronizar formatos e resolver inconsistências entre diferentes fontes de dados (ex: CRM antigo e novo ERP) durante processos de migração. **E-commerce e Catálogos de Produtos**: Corrigir erros de digitação, padronizar nomes e descrições de produtos e categorizar itens automaticamente para melhorar a experiência do cliente e a gestão de estoque. **Setor Financeiro e Regulatório**: Garantir a conformidade de dados (ex: KYC - Know Your Customer) através da identificação e correção de registros duplicados ou incompletos [1].

## Pitfalls
**Vagueza e Ambiguidade**: Prompts como "Corrija o formato" ou "Remova dados ruins" são muito vagos e levam a resultados incorretos ou incompletos [1]. **Ignorar Limites de Contexto**: Tentar processar conjuntos de dados muito grandes de uma só vez pode exceder o limite de tokens do LLM, resultando em truncamento ou falha [2]. **Confiança Excessiva na IA**: A IA pode introduzir novos erros (alucinações) ou aplicar a lógica de limpeza de forma incorreta. A validação humana e a revisão dos resultados são cruciais [2]. **Não Especificar o Formato de Saída**: Falhar em pedir o resultado em um formato estruturado (ex: CSV, JSON, ou apenas a lista de correções) pode dificultar a aplicação das mudanças de volta ao conjunto de dados original. **Não Fornecer Contexto Suficiente**: Para tarefas complexas (ex: preenchimento de valores ausentes), a IA precisa de contexto sobre as colunas vizinhas e o tipo de dados para fazer inferências precisas.

## URL
[https://numerous.ai/blog/ai-prompts-for-data-cleaning](https://numerous.ai/blog/ai-prompts-for-data-cleaning)
