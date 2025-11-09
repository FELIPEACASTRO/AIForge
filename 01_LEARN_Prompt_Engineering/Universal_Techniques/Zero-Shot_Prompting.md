# Zero-Shot Prompting

## Description
O **Zero-Shot Prompting** (ou "Prompting de Disparo Zero") é a forma mais fundamental e direta de interagir com um Large Language Model (LLM). A técnica consiste em fornecer ao modelo uma instrução ou pergunta para realizar uma tarefa específica, sem incluir exemplos prévios (demonstrações) de pares de entrada e saída. O modelo deve confiar inteiramente em seu conhecimento interno, adquirido durante o pré-treinamento em vastos conjuntos de dados, e em sua capacidade de seguir instruções (aprimorada por técnicas como *Instruction Tuning* e *RLHF* - Reinforcement Learning from Human Feedback) para gerar a resposta apropriada. É a abordagem padrão e mais simples, ideal para tarefas bem definidas e para modelos modernos que já possuem alta capacidade de generalização.

## Examples
```
1. **Classificação de Sentimento:**
   ```
   Classifique o sentimento do seguinte texto como "Positivo", "Negativo" ou "Neutro".
   Texto: "Apesar do atraso, o atendimento foi impecável e o produto superou minhas expectativas."
   Sentimento:
   ```

2. **Extração de Entidades:**
   ```
   Extraia o nome da pessoa, a organização e a localização deste texto.
   Texto: "A Dra. Ana Silva, CEO da TechSolutions, fará uma palestra em São Paulo na próxima semana."
   Pessoa:
   Organização:
   Localização:
   ```

3. **Tradução Simples:**
   ```
   Traduza a seguinte frase para o português.
   Frase: "The quick brown fox jumps over the lazy dog."
   Tradução:
   ```

4. **Resumo de Texto:**
   ```
   Resuma o parágrafo abaixo em uma única frase.
   Parágrafo: "A energia solar fotovoltaica é a principal fonte de energia renovável em crescimento no mundo. Ela converte a luz do sol diretamente em eletricidade, utilizando células fotovoltaicas, e tem um impacto ambiental significativamente menor do que os combustíveis fósseis."
   Resumo:
   ```

5. **Geração de Código (Função Simples):**
   ```
   Escreva uma função em Python que calcule o fatorial de um número inteiro positivo.
   ```

6. **Resposta a Perguntas (Factual):**
   ```
   Qual é a capital do Canadá e qual é o seu idioma oficial?
   ```

7. **Reescrita de Estilo:**
   ```
   Reescreva a seguinte frase em um tom mais formal e profissional.
   Frase: "A gente precisa dar um jeito de terminar isso logo, tá?"
   Reescrita:
   ```
```

## Best Practices
*   **Seja Explícito e Claro:** A instrução deve ser o mais clara e detalhada possível, definindo a tarefa, o formato de saída e quaisquer restrições.
*   **Use Delimitadores:** Para prompts mais longos ou com dados de entrada, use delimitadores (como `###`, `"""`, ou tags XML) para separar a instrução do contexto ou dos dados.
*   **Especifique o Formato de Saída:** Peça explicitamente o formato desejado (e.g., "Responda em formato JSON", "Liste em tópicos", "Apenas a palavra-chave").
*   **Instruções Negativas (Evitar):** Evite dizer ao modelo o que *não* fazer. Em vez disso, diga o que *deve* ser feito. Por exemplo, em vez de "Não inclua a introdução", diga "Comece diretamente com o primeiro ponto".
*   **Modelos Modernos:** Utilize modelos mais recentes e aprimorados (*Instruction-Tuned*), pois eles são inerentemente mais eficazes no Zero-Shot Prompting.

## Use Cases
*   **Classificação Rápida:** Classificação de e-mails, tickets de suporte ou comentários de clientes em categorias pré-definidas (e.g., urgência, tópico, sentimento).
*   **Extração de Dados:** Extrair informações específicas (nomes, datas, valores) de documentos ou textos não estruturados.
*   **Tradução e Resumo:** Tarefas simples de tradução de frases ou resumos concisos onde a precisão contextual extrema não é crítica.
*   **Geração de Conteúdo Inicial:** Criar rascunhos, títulos, ou esboços de artigos e códigos para tarefas simples e diretas.
*   **Testes de Capacidade do Modelo:** Avaliar rapidamente a capacidade de generalização de um novo LLM em uma variedade de tarefas.

## Pitfalls
*   **Ambiguidade:** Instruções vagas ou ambíguas levam a resultados inconsistentes ou incorretos. O modelo não tem exemplos para inferir a intenção.
*   **Tarefas Complexas:** Não é adequado para tarefas que exigem raciocínio multi-etapas, planejamento ou conhecimento muito específico e não comum (onde Few-Shot ou Chain-of-Thought seriam melhores).
*   **Dependência Excessiva:** Confiar demais na capacidade de generalização do modelo para tarefas onde a formatação ou o estilo são cruciais.
*   **Ausência de Contexto:** Não fornecer o contexto necessário para a tarefa, assumindo que o modelo "sabe" o que você quer.
*   **Alucinações:** Em tarefas de geração de fatos ou dados, a ausência de exemplos ou contexto pode aumentar a probabilidade de o modelo "alucinar" informações.

## URL
[https://www.promptingguide.ai/techniques/zeroshot](https://www.promptingguide.ai/techniques/zeroshot)
