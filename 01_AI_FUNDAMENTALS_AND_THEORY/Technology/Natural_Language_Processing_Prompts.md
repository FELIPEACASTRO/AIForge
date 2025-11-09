# Natural Language Processing Prompts

## Description
Prompts de Processamento de Linguagem Natural (PLN) referem-se à arte e ciência de criar instruções de entrada (prompts) para modelos de linguagem grandes (LLMs) com o objetivo de realizar tarefas específicas de PLN. Em vez de treinar um modelo para cada tarefa (como classificação de texto, sumarização ou tradução), a engenharia de prompt permite que um único LLM seja adaptado para uma vasta gama de aplicações através de instruções textuais. Esta técnica é fundamental para extrair o máximo de valor dos LLMs, garantindo que a saída seja precisa, relevante e formatada conforme a necessidade. O foco principal é fornecer contexto, definir o papel do modelo, especificar o formato de saída e, para tarefas complexas, guiar o raciocínio do modelo (como no Chain-of-Thought).

## Examples
```
**1. Classificação de Sentimento (Few-Shot):**
```
Você é um classificador de sentimento. Classifique o texto como 'Positivo', 'Negativo' ou 'Neutro'.

Exemplo 1:
Texto: O serviço foi rápido e a comida estava excelente.
Sentimento: Positivo

Exemplo 2:
Texto: Atrasou, mas o produto chegou intacto.
Sentimento: Neutro

Texto: O atendimento foi péssimo e o problema não foi resolvido.
Sentimento:
```

**2. Sumarização Extrativa (Com Restrição de Formato):**
```
Sumarize o texto abaixo em 3 frases, extraindo apenas as informações mais críticas. A saída deve ser uma lista numerada.

[TEXTO LONGO AQUI]
```

**3. Extração de Entidades Nomeadas (NER) (Com Saída Estruturada):**
```
Extraia todas as entidades de 'Pessoa', 'Organização' e 'Local' do texto a seguir. A saída deve estar no formato JSON.

Texto: Maria Silva, CEO da TechCorp, viajou para Paris para a conferência de IA.

JSON:
```

**4. Geração de Código a partir de Intenção (Com Contexto de Linguagem):**
```
Você é um assistente de programação Python. Gere o código Python para a seguinte tarefa:

Tarefa: Criar uma função que recebe uma lista de números e retorna a média, ignorando valores nulos.

Código Python:
```

**5. Tradução com Adaptação de Estilo (Com Definição de Público):**
```
Traduza o seguinte parágrafo do Português para o Inglês. O tom deve ser formal e o público-alvo são executivos de alto nível.

Parágrafo: A implementação da nova política de governança de dados é crucial para a conformidade regulatória e para a mitigação de riscos operacionais.

Tradução:
```
```

## Best Practices
**1. Seja Específico e Contextualizado:** Defina claramente o papel do modelo (ex: "Você é um analista financeiro experiente"), o público-alvo e o formato de saída desejado (JSON, lista, parágrafo). **2. Use o Chain-of-Thought (CoT):** Para tarefas complexas de raciocínio (como análise de risco ou resolução de problemas), instrua o modelo a "pensar passo a passo" antes de fornecer a resposta final. **3. Forneça Exemplos (Few-Shot):** Inclua 1 a 3 exemplos de pares de entrada/saída para orientar o modelo sobre o estilo, tom e estrutura da resposta esperada. **4. Isole a Tarefa de PLN:** Se a tarefa puder ser resolvida com métodos tradicionais de PLN (como regex ou contagem), use-os. Caso contrário, quebre a tarefa complexa em subtarefas, usando o LLM apenas para as partes que exigem compreensão de linguagem natural. **5. Itere e Refine:** Comece com um prompt simples e adicione restrições, contexto e exemplos conforme necessário para melhorar a qualidade da saída. **6. Evite Negações:** Em vez de dizer "Não inclua a introdução", diga "Comece diretamente com a seção de resultados". Modelos tendem a processar melhor instruções positivas.

## Use Cases
**1. Análise de Sentimento e Classificação de Texto:** Classificar avaliações de clientes, e-mails de suporte ou notícias em categorias predefinidas (ex: positivo, negativo, spam, urgência). **2. Sumarização e Geração de Conteúdo:** Criar resumos de documentos longos (artigos, relatórios financeiros), gerar títulos ou descrições de produtos. **3. Extração de Informação (NER e Relações):** Identificar e extrair entidades nomeadas (pessoas, locais, datas, valores) e as relações entre elas em textos não estruturados (ex: contratos, prontuários médicos). **4. Tradução e Adaptação de Estilo:** Traduzir textos entre idiomas, adaptando o tom (formal/informal) ou o jargão para um público específico (ex: jurídico, técnico). **5. Geração de Código e Documentação:** Auxiliar desenvolvedores na criação de snippets de código, documentação técnica ou na explicação de funções complexas. **6. Chatbots e Assistentes Virtuais:** Melhorar a compreensão da intenção do usuário e a geração de respostas contextuais em sistemas de conversação.

## Pitfalls
**1. Prompts Vagos ou Ambíguos:** Não especificar o objetivo, o formato ou o público-alvo leva a respostas genéricas e de baixa qualidade. **2. Confiar em Role-Prompting Excessivo:** Atribuir um papel ("Você é um especialista...") sem fornecer contexto ou restrições claras pode não melhorar o desempenho e apenas aumentar o custo (mais tokens). **3. Usar LLMs para Tarefas Simples de Programação:** Tentar usar o LLM para tarefas que podem ser resolvidas de forma mais eficiente e confiável com código simples (ex: cálculos, regex, manipulação de strings) resulta em desperdício de recursos e maior latência. **4. Ignorar a Necessidade de Raciocínio:** Para tarefas que exigem múltiplas etapas de lógica, não incluir instruções como "pense passo a passo" (CoT) pode levar a erros de raciocínio ou "alucinações". **5. Falha em Validar a Saída:** Assumir que a saída do LLM está sempre correta, especialmente em tarefas de extração de dados ou fatos, sem um mecanismo de validação ou verificação. **6. Prompt Injection:** Não proteger o prompt contra entradas maliciosas que tentam desviar o modelo de sua tarefa original.

## URL
[https://www.promptingguide.ai/papers](https://www.promptingguide.ai/papers)
