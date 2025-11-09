# Prompts de Classificação de Texto

## Description
A Classificação de Texto é uma técnica fundamental de Processamento de Linguagem Natural (PLN) que envolve a categorização de um texto em uma ou mais classes ou rótulos predefinidos. No contexto de Large Language Models (LLMs), os **Prompts de Classificação de Texto** são instruções cuidadosamente elaboradas que orientam o modelo a realizar essa tarefa de categorização. Em vez de treinar um modelo de aprendizado de máquina tradicional, o LLM é instruído, via prompt, a atuar como um classificador. A eficácia reside na clareza e precisão com que as categorias e o formato de saída são definidos. Técnicas como *Zero-Shot* (sem exemplos), *Few-Shot* (com poucos exemplos) e a especificação de um formato de saída estruturado (como JSON ou uma única palavra) são cruciais para garantir resultados consistentes e utilizáveis em fluxos de trabalho de dados [1] [2]. A principal vantagem é a capacidade de realizar classificações complexas sem a necessidade de grandes conjuntos de dados rotulados para treinamento, aproveitando o conhecimento pré-treinado do LLM.

## Examples
```
**1. Classificação de Sentimento (Zero-Shot):**
```
Classifique o texto a seguir em uma das seguintes categorias: Positivo, Negativo ou Neutro. Retorne APENAS a categoria.
Texto: "O atendimento foi rápido, mas o produto veio com defeito."
Categoria:
```

**2. Classificação de Intenção (Few-Shot):**
```
Você é um classificador de intenções de clientes. Classifique a solicitação do cliente em uma das seguintes intenções: 'Suporte Técnico', 'Consulta de Faturamento', 'Cancelamento de Serviço'. Se não se encaixar, use 'Outro'.

Exemplo 1:
Solicitação: "Minha internet parou de funcionar e preciso de ajuda urgente."
Intenção: Suporte Técnico

Exemplo 2:
Solicitação: "Gostaria de saber o valor da minha próxima fatura."
Intenção: Consulta de Faturamento

Solicitação: "Quero encerrar minha conta e parar de receber cobranças."
Intenção:
```

**3. Classificação de Tópico com Saída Estruturada (JSON):**
```
Classifique o artigo de notícias a seguir em um único tópico principal. Os tópicos permitidos são: 'Política', 'Economia', 'Esportes', 'Tecnologia', 'Saúde'.
Retorne o resultado em formato JSON com as chaves "id_topico" (inteiro) e "nome_topico" (string).
Artigo: "A bolsa de valores atingiu um novo recorde após a divulgação dos dados de inflação."
JSON:
```

**4. Classificação de Gravidade de Ticket (Escala Numérica):**
```
Classifique a gravidade do ticket de suporte em uma escala de 1 a 5, onde 1 é 'Baixa' e 5 é 'Crítica'. Retorne APENAS o número.
Ticket: "O servidor de e-mail está lento, mas ainda funcional para a maioria dos usuários."
Gravidade:
```

**5. Classificação de Relevância de Documento (Booleana):**
```
Determine se o parágrafo a seguir é relevante para o tema 'Regulamentação de IA'. Responda APENAS com 'Sim' ou 'Não'.
Parágrafo: "O novo projeto de lei visa estabelecer diretrizes éticas para o desenvolvimento e uso de sistemas de inteligência artificial em setores públicos."
Relevante:
```
```

## Best Practices
**1. Definição Clara das Categorias:** MUST definir as categorias de forma inequívoca e mutuamente exclusiva. Fornecer uma breve descrição de cada categoria dentro do prompt.
**2. Especificação do Formato de Saída:** Sempre instruir o LLM a retornar a classificação em um formato estruturado (ex: JSON, uma única palavra, ou um rótulo específico) para facilitar o processamento automatizado (*downstream*).
**3. Uso de Few-Shot Prompting:** Para tarefas mais complexas ou categorias específicas do domínio, incluir 2 a 5 exemplos de pares (texto de entrada, rótulo de saída) para orientar o modelo.
**4. Instruções de Saída Única:** Pedir ao modelo para retornar *apenas* o rótulo da categoria, sem explicações ou texto adicional, a menos que a explicação seja explicitamente solicitada.
**5. Tratamento de Ambiguidade/Outros:** Incluir uma categoria "Outro" ou "Não Aplicável" e instruir o modelo a usá-la quando o texto não se encaixar claramente nas categorias definidas.

## Use Cases
**1. Análise de Sentimento:** Classificar avaliações de produtos, comentários em mídias sociais ou feedback de clientes como positivo, negativo ou neutro.
**2. Roteamento de Tickets de Suporte:** Classificar automaticamente tickets de suporte ou e-mails de clientes para roteá-los para o departamento correto (ex: Faturamento, Técnico, Vendas).
**3. Filtragem de Conteúdo:** Classificar conteúdo gerado por usuários ou artigos de notícias em categorias como *spam*, *conteúdo impróprio*, *fake news* ou tópicos específicos (ex: Esportes, Política).
**4. Classificação de Documentos:** Categorizar documentos jurídicos, médicos ou empresariais (ex: Contrato, Fatura, Relatório) para organização e recuperação de informações.
**5. Análise de Intenção de Compra:** Classificar interações de *chatbots* ou transcrições de chamadas para determinar a intenção do cliente (ex: *Interesse*, *Dúvida*, *Pronto para Comprar*).

## Pitfalls
**1. Categorias Ambíguas:** Definir categorias que se sobrepõem ou não são claras, levando a classificações inconsistentes ou erradas pelo LLM.
**2. Falta de Restrição de Saída:** Não especificar o formato de saída, resultando em respostas longas, não estruturadas e difíceis de processar automaticamente.
**3. Viés do Modelo:** O LLM pode introduzir vieses presentes em seus dados de treinamento, afetando a classificação de textos sensíveis ou de grupos minoritários.
**4. Overfitting no Few-Shot:** Usar exemplos de *Few-Shot* que são muito específicos ou que não representam a diversidade do conjunto de dados real, limitando a capacidade de generalização do modelo.
**5. Ignorar o Contexto:** Não fornecer contexto suficiente sobre o domínio ou a tarefa, fazendo com que o LLM use seu conhecimento geral em vez das regras de classificação específicas.
**6. Prompt Longo Demais:** Incluir muitas categorias ou exemplos, o que pode exceder o limite de contexto do modelo ou diluir a instrução principal.

## URL
[https://www.promptingguide.ai/prompts/classification](https://www.promptingguide.ai/prompts/classification)
