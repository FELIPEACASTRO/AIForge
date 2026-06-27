# Instruction Prompting (Prompting de Instrução)

## Description
**Instruction Prompting** (Prompting de Instrução) é a técnica fundamental e mais básica da Engenharia de Prompt, centrada na capacidade dos Modelos de Linguagem Grande (LLMs) de seguir diretivas expressas em linguagem natural. Consiste em fornecer comandos claros e concisos ao modelo para que ele execute uma tarefa específica. Diferentemente de técnicas mais avançadas que exigem exemplos (Few-Shot) ou raciocínio em cadeia (Chain-of-Thought), o Instruction Prompting se baseia puramente na instrução para guiar o modelo a realizar tarefas novas ou não vistas anteriormente, sem a necessidade de treinamento específico ou grandes conjuntos de dados rotulados. É a base para a comunicação eficaz com a IA, permitindo que os usuários transformem tarefas complexas (como formatação de texto, extração de dados ou avaliação) em comandos simples e escaláveis.

## Examples
```
**1. Extração e Formatação de Dados**

**Prompt:**
```
Leia o texto a seguir e extraia o nome completo, o cargo e a empresa. Formate a saída como um objeto JSON.

Texto: "Prezado Sr. João Silva, como Gerente de Projetos da TechSolutions, gostaria de agendar uma reunião."
```

**2. Resumo Condicional**

**Prompt:**
```
Resuma o seguinte artigo em português. O resumo deve ter no máximo 100 palavras e ser escrito em um tom formal e objetivo.

Artigo: [Insira o texto do artigo aqui]
```

**3. Classificação e Categorização**

**Prompt:**
```
Classifique a seguinte frase em uma das categorias: [Vendas], [Suporte Técnico], [Faturamento] ou [Geral].

Frase: "Minha fatura deste mês parece incorreta, preciso de ajuda para verificar os valores."
```

**4. Geração de Código com Restrições**

**Prompt:**
```
Escreva uma função em Python chamada 'calcular_imc' que receba dois argumentos (peso em kg e altura em metros) e retorne o Índice de Massa Corporal (IMC). Não inclua comentários no código.
```

**5. Tradução e Adaptação de Tom**

**Prompt:**
```
Traduza o seguinte parágrafo do inglês para o português. O tom da tradução deve ser informal e amigável.

Parágrafo: "The prompt engineering technique is essential for maximizing the utility of large language models."
```

**6. Avaliação e Feedback Estruturado**

**Prompt:**
```
Avalie o seguinte trecho de texto com base em dois critérios: 'Gramática' e 'Clareza'. Atribua uma pontuação de 1 a 10 para cada critério e forneça um breve raciocínio para a pontuação.

Texto: "Apesar da crença popular, não há evidências sólidas que suportem a ideia de que jogos de vídeo levam a comportamento violento."
```
```

## Best Practices
**Clareza e Especificidade:** Use verbos de ação diretos (e.g., "Escreva", "Analise", "Compare") e evite ambiguidades. Quanto mais específico o comando, melhor o resultado. **Separação de Instrução e Contexto:** Utilize delimitadores (como `###`, `"""`, ou tags XML) para separar claramente a instrução principal do contexto ou dos dados de entrada. **Instruções no Início:** Coloque a instrução mais importante no início do prompt para garantir que o modelo a priorize. **Iteração e Refinamento:** Comece com instruções simples e refine-as progressivamente com base nas saídas do modelo. **Evitar Negações:** Concentre-se no que o modelo deve fazer, e não no que ele não deve fazer. Por exemplo, use "Resuma em 50 palavras" em vez de "Não escreva mais de 50 palavras".

## Use Cases
**Processamento de Dados:** Formatação e extração de dados estruturados (e.g., nomes, endereços, datas) de texto não estruturado. **Redação e Edição:** Geração de conteúdo com restrições específicas de formato, tom, tamanho ou estilo (e.g., e-mails formais, posts curtos para redes sociais). **Classificação e Rótulo:** Atribuição de categorias ou rótulos a textos (e.g., classificação de tickets de suporte, análise de sentimento). **Revisão e Feedback Automatizado:** Avaliação de textos (e.g., redações, resumos) com base em critérios definidos, fornecendo pontuações e justificativas. **Remoção de Informações Sensíveis (PII):** Identificação e substituição automática de dados pessoais (e.g., nomes, telefones, e-mails) em documentos para fins de privacidade.

## Pitfalls
**Ambiguidade e Imprecisão:** Usar linguagem vaga ou termos com múltiplos significados. O modelo pode interpretar a instrução de forma diferente da intenção do usuário. **Instruções Conflitantes:** Incluir comandos que se contradizem no mesmo prompt (e.g., "Seja conciso, mas detalhe todos os pontos"). **Sobrecarga de Informação:** Tentar incluir muitas tarefas complexas em uma única instrução, o que pode levar o modelo a ignorar partes do comando. **Falta de Delimitadores:** Não separar a instrução dos dados de entrada, fazendo com que o modelo confunda o que é comando e o que é contexto. **Confiança Excessiva:** Assumir que o modelo entenderá o contexto ou a intenção sem que isso seja explicitamente declarado. **Uso de Negações:** Dizer ao modelo o que **não** fazer (e.g., "Não use jargão") é menos eficaz do que dizer o que **fazer** (e.g., "Use linguagem simples").

## URL
[https://learnprompting.org/docs/basics/instructions](https://learnprompting.org/docs/basics/instructions)
