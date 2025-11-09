# Prompt Ensembling

## Description

Prompt Ensembling é uma técnica de engenharia de prompt que visa aumentar a **confiabilidade** e a **precisão** das respostas de Modelos de Linguagem Grande (LLMs) ao combinar as saídas de múltiplos *prompts* diferentes para a mesma questão. Em vez de depender de uma única formulação de *prompt*, o *ensembling* gera diversas perspectivas sobre o problema e agrega as respostas para chegar a um resultado final mais robusto. As principais abordagens incluem a diversificação dos *prompts* (por exemplo, alterando os exemplos *Few-Shot* ou a formulação da pergunta) e a agregação das respostas (por exemplo, por votação majoritária ou por um verificador treinado). O objetivo é reduzir o viés na distribuição de saída do LLM e mitigar erros que surgiriam de um único *prompt* mal formulado.

## Statistics

- **DiVeRSe (Li et al., 2022):** Demonstrou melhorias na confiabilidade, especialmente em tarefas de raciocínio como o *benchmark* GSM8K, ao usar 5 *prompts* diferentes e amostrar 20 caminhos de raciocínio para cada um (totalizando 100 conclusões).
- **AMA (Arora et al., 2022):** Foi capaz de fazer com que um modelo menor (GPT-J-6B) superasse o desempenho do GPT-3 em questões onde o contexto fornecido continha a resposta, usando a estratégia de geração de múltiplas perguntas e agregação ponderada.
- **Uso Geral:** O *ensembling* é frequentemente citado como uma técnica fundamental para aumentar a robustez e a precisão em tarefas críticas.

## Features

- **Aumento da Confiabilidade:** Reduz a variância e o viés nas respostas do LLM.
- **Diversificação de Prompts:** Utiliza múltiplas formulações de *prompt* para o mesmo problema.
- **Agregação de Respostas:** Combina as saídas diversas através de métodos como votação majoritária ou verificadores treinados.
- **Técnicas Específicas:** Inclui métodos como **DiVeRSe** (Diverse Verifier on Reasoning Steps), que varia os exemplos *Few-Shot* e usa um verificador de votação, e **AMA** (Ask Me Anything), que usa um LLM para gerar variações da pergunta e aplica agregação ponderada.

## Use Cases

- **Aumento da Precisão em Tarefas de Raciocínio:** Melhoria na resolução de problemas matemáticos e lógicos (como demonstrado pelo DiVeRSe no GSM8K).
- **Melhoria na Extração de Informações:** Aumentar a acurácia na extração de fatos de um contexto fornecido (como demonstrado pelo AMA).
- **Sistemas de Perguntas e Respostas (Q&A):** Obter respostas mais confiáveis em sistemas que dependem da precisão do LLM.
- **Controle de Qualidade e Segurança:** Usado para verificar a consistência das saídas do LLM, sendo um componente chave em estratégias de confiabilidade (Reliability) e autoavaliação (Self-Evaluation) de LLMs.

## Integration

A estratégia mais simples e recomendada na prática é a **votação majoritária** (Self-Consistency), onde a resposta mais frequente entre as diversas saídas é escolhida.

**Exemplo de Aplicação (Votação Majoritária):**

1.  **Gere Múltiplos Prompts:** Crie 5 a 10 variações do seu *prompt* original (ex: altere a *persona*, a formatação, ou os exemplos *Few-Shot*).
2.  **Obtenha Múltiplas Respostas:** Execute cada *prompt* no LLM.
3.  **Agregue por Votação:** Conte a frequência de cada resposta final.

*Prompt* de Exemplo (para gerar variações):
`"Contexto: [Texto]. Tarefa: Classifique o sentimento do texto como Positivo, Negativo ou Neutro. Gere 5 variações desta pergunta, mantendo o contexto e a tarefa, mas alterando a formulação."`

*Prompt* de Exemplo (para agregação):
`"Dadas as seguintes respostas: [Lista de Respostas]. Qual é a resposta mais provável/correta? Justifique sua escolha."` (Embora a votação majoritária seja mais simples, um *prompt* de agregação pode ser usado para tarefas mais complexas).

## URL

https://learnprompting.org/docs/reliability/ensembling