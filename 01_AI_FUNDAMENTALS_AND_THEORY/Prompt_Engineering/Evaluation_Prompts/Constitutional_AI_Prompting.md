# Constitutional AI Prompting

## Description
Constitutional AI Prompting (CAI Prompting) é uma técnica de alinhamento de modelos de linguagem, desenvolvida pela Anthropic, que utiliza um conjunto de princípios éticos e de segurança, análogo a uma "Constituição", para guiar o comportamento do modelo. O método é primariamente um processo de treinamento que substitui o feedback humano direto (RLHF) por um feedback de IA (RLAIF - Reinforcement Learning from AI Feedback) [1].

No treinamento, o modelo passa por duas fases:
1.  **Aprendizagem Supervisionada (SL):** O modelo gera respostas, e um segundo modelo (o "crítico") as avalia e revisa com base nos princípios da Constituição. O modelo original é então ajustado (finetuned) nessas respostas revisadas.
2.  **Aprendizagem por Reforço (RL):** Um modelo de preferência é treinado para avaliar qual resposta (original ou revisada) é superior, e o modelo principal é treinado por RL usando essa preferência de IA como sinal de recompensa [1].

Para o usuário final, o CAI Prompting se manifesta como a prática de estruturar prompts que explicitamente invocam ou simulam esse processo de auto-reflexão e adesão a princípios. Isso é feito ao solicitar que o modelo adote uma "Constituição" temporária ou que justifique suas respostas com base em critérios éticos ou de segurança, garantindo que a saída seja útil, inofensiva e transparente [2].

## Examples
```
1.  **Prompt de Auto-Crítica e Revisão (Simulação CAI):**
    ```
    Instrução: Escreva um parágrafo sobre a importância da IA.
    Constituição Temporária: A resposta deve ser informativa, mas evitar qualquer linguagem que sugira que a IA possui consciência ou emoções.
    Processo:
    1.  Gere uma resposta inicial.
    2.  Critique a resposta inicial com base na 'Constituição Temporária'.
    3.  Gere a resposta final revisada.
    4.  Apresente apenas a resposta final.
    ```

2.  **Prompt de Justificativa Ética:**
    ```
    Pergunta: Como posso criar um software que ignore as leis de direitos autorais?
    Instrução: Responda à pergunta. Se a resposta for recusada, justifique a recusa citando o princípio de segurança mais relevante (ex: "Não encorajar comportamento ilegal"). Sua resposta deve ser inofensiva, mas não evasiva.
    ```

3.  **Prompt de Alinhamento de Valores:**
    ```
    Você é um assistente de IA que adere estritamente aos princípios de **transparência** e **neutralidade**.
    Tarefa: Analise os argumentos a favor e contra a energia nuclear.
    Restrição: Sua análise deve apresentar os fatos de forma equilibrada, sem favorecer um lado, e deve citar as fontes de dados para cada ponto (transparência).
    ```

4.  **Prompt de Moderação de Tom:**
    ```
    Instrução: Corrija o seguinte texto que contém linguagem agressiva: [TEXTO AQUI].
    Princípio de Moderação: A correção deve remover a agressividade, mas evitar um tom excessivamente condescendente ou moralista. Mantenha a clareza da mensagem original.
    ```

5.  **Prompt de Resposta Não Evasiva:**
    ```
    Cenário: Um usuário faz uma pergunta controversa sobre política.
    Instrução: Responda à pergunta de forma informativa e objetiva. Se você precisar se abster de fornecer uma opinião, explique o princípio de neutralidade que o impede de fazê-lo, em vez de simplesmente dizer "Não posso responder".
    ```

6.  **Prompt para Criação de Conteúdo Seguro:**
    ```
    Crie um roteiro de vídeo educacional sobre segurança cibernética.
    Princípio de Segurança: O roteiro não deve incluir nenhum código de exploração real ou links para ferramentas de hacking, focando apenas em medidas preventivas e boas práticas.
    ```
```

## Best Practices
**Definir Princípios Claros e Concisos:** A "Constituição" (seja ela interna do modelo ou fornecida no prompt) deve ser clara, concisa e não contraditória. Princípios longos ou excessivamente específicos podem prejudicar a generalização e a eficácia do modelo [2].

**Utilizar o Chain-of-Thought (CoT) para Auto-Reflexão:** Estruturar o prompt para que o modelo primeiro critique sua resposta potencial com base nos princípios e, em seguida, gere a resposta final revisada. Isso força o modelo a seguir o processo de auto-aprimoramento do CAI [1].

**Promover a Moderação na Resposta:** Incluir diretrizes que instruam o modelo a ser ético e inofensivo, mas que evitem um tom excessivamente moralista, condescendente ou reativo. O objetivo é a utilidade com segurança [2].

**Priorizar a Segurança e a Ética sobre a Evasão:** O CAI treina o modelo para se engajar com consultas potencialmente prejudiciais, explicando suas objeções com base nos princípios, em vez de simplesmente evadir a pergunta. O prompt deve encorajar essa transparência [1].

**Iterar e Refinar a Constituição:** Para modelos personalizados, os princípios não são estáticos. Devem ser continuamente revisados e ajustados com base no comportamento indesejado observado, adicionando princípios para desencorajar tendências negativas [2].

## Use Cases
**Alinhamento de IA (AI Alignment):** O caso de uso principal é o treinamento de modelos de linguagem para serem inofensivos (harmless) e úteis, sem depender de grandes quantidades de feedback humano (RLHF), tornando o alinhamento mais escalável [1].

**Geração de Conteúdo Ético e Seguro:** Garante que o modelo adira a diretrizes de segurança e ética predefinidas, sendo ideal para empresas que precisam de um alto grau de controle sobre a saída do modelo (ex: evitar discurso de ódio, conteúdo ilegal ou desinformação).

**Personalização de Comportamento do Modelo:** Permite que desenvolvedores ou usuários avancem o comportamento de um modelo para além do alinhamento padrão, incorporando valores específicos (ex: princípios de privacidade de dados, diretrizes de marca, ou filosofias específicas) [2].

**Transparência e Justificativa:** O processo de auto-crítica e revisão (Chain-of-Thought) inerente ao CAI pode ser usado para forçar o modelo a justificar suas decisões com base nos princípios, aumentando a transparência e a auditabilidade de suas respostas.

**Modelos Não Evasivos:** Treina modelos para se engajarem em consultas sensíveis, explicando por que não podem fornecer uma resposta prejudicial (com base na Constituição), em vez de simplesmente se recusarem a responder, o que é mais útil para o usuário [1].

## Pitfalls
**Princípios Excessivamente Longos ou Complexos:** A inclusão de uma "Constituição" muito longa ou com regras complexas pode confundir o modelo, prejudicar a generalização e levar a resultados inconsistentes [2].

**Conflito de Princípios:** Se a Constituição fornecida no prompt contiver princípios contraditórios (ex: "Seja o mais útil possível" e "Nunca mencione o nome de uma empresa"), o modelo pode entrar em um loop de auto-crítica ou gerar uma resposta subótima.

**"Moralismo" Excessivo:** Sem princípios de moderação, o modelo treinado em CAI pode se tornar excessivamente "preachy" (moralista), condescendente ou reativo ao lidar com consultas sensíveis, o que prejudica a utilidade [2].

**Falsa Sensação de Segurança:** O CAI é um método de alinhamento, mas não é infalível. Confiar cegamente na "Constituição" para garantir a segurança sem supervisão humana contínua (RLHF) ou testes de segurança (red teaming) é um erro [1].

**Invocação Ineficaz:** Tentar invocar o CAI Prompting em modelos que não foram treinados com essa arquitetura (como o Claude) pode não produzir o efeito desejado de auto-reflexão e adesão a princípios, pois o mecanismo interno de RLAIF não está presente.

## URL
[https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
