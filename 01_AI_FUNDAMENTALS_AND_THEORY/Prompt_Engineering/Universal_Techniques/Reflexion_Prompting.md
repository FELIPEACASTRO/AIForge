# Reflexion

## Description
A técnica **Reflexion** é um *framework* inovador que aprimora a capacidade de Agentes de Linguagem (LLM Agents) de resolver tarefas complexas através de um processo de **auto-reflexão** e **reforço verbal**. Em vez de ajustar os pesos do modelo (como no aprendizado por reforço tradicional), o Reflexion utiliza o próprio LLM para gerar *feedback linguístico* sobre suas tentativas anteriores, transformando esse feedback em uma memória dinâmica e um guia para a próxima iteração.

O processo é tipicamente iterativo e envolve:
1. **Tentativa de Ação:** O agente tenta resolver a tarefa (e.g., gerar código, responder a uma pergunta complexa, planejar uma sequência de ações).
2. **Observação e Avaliação:** O agente recebe um resultado (e.g., um erro de compilação, uma resposta incorreta, um *feedback* do ambiente).
3. **Reflexão:** O agente usa o *feedback* e o histórico da tentativa para gerar uma **reflexão** crítica. Esta reflexão é uma análise textual que identifica as falhas, os pontos de melhoria e as novas estratégias a serem adotadas.
4. **Nova Tentativa:** A reflexão gerada é adicionada à memória de contexto do agente e usada para guiar a próxima tentativa de ação, resultando em um desempenho progressivamente melhorado.

O Reflexion pode ser aplicado em conjunto com outras técnicas, como o ReAct (Reasoning and Acting), para criar agentes mais robustos e autônomos.

## Examples
```
1.  **Exemplo 1: Geração de Código (Tentativa Inicial)**
    *   **Prompt (Usuário):** "Gere uma função em Python que calcule o n-ésimo número de Fibonacci de forma recursiva e otimizada com memoização."
    *   **Resposta do LLM (Tentativa 1):** (Código com erro de lógica ou sem memoização)
    *   **Prompt (Reflexão):** "A tentativa anterior falhou ao incluir a memoização, resultando em complexidade de tempo O(2^n). Analise o código, identifique onde a memoização deve ser aplicada (e.g., usando um dicionário ou `functools.lru_cache`) e gere uma nova função que atenda ao requisito de otimização."

2.  **Exemplo 2: Resolução de Problemas (Tentativa Inicial)**
    *   **Prompt (Usuário):** "Um trem viaja a 60 km/h. Se ele percorrer 300 km, quanto tempo levará? Apresente o raciocínio passo a passo."
    *   **Resposta do LLM (Tentativa 1):** (Raciocínio correto, mas erro de cálculo final, e.g., 300/60 = 6 horas)
    *   **Prompt (Reflexão):** "A sua resposta anterior continha um erro de cálculo: 300 dividido por 60 é 5, não 6. Revise o passo a passo do raciocínio, corrija o cálculo e forneça a resposta final correta. A reflexão deve ser: 'O erro foi na divisão final. 300/60 = 5. A resposta correta é 5 horas.'"

3.  **Exemplo 3: Interação com Ambiente (Agente de API)**
    *   **Prompt (Usuário):** "Busque o preço atual do Bitcoin na API da CoinGecko e converta para Reais (BRL)."
    *   **Resposta do LLM (Tentativa 1):** (Falha na chamada da API devido a um parâmetro incorreto, e.g., `moeda=Real` em vez de `moeda=brl`)
    *   **Prompt (Reflexão):** "A chamada de API na tentativa anterior retornou um erro 400 (Bad Request) com a mensagem 'Parâmetro de moeda inválido'. O parâmetro 'Real' não é aceito. Consulte a documentação da API e ajuste o parâmetro para o código ISO correto da moeda brasileira (BRL). Repita a chamada e forneça o resultado."

4.  **Exemplo 4: Escrita Criativa e Estilística**
    *   **Prompt (Usuário):** "Escreva um parágrafo de abertura para um conto de ficção científica no estilo de H.P. Lovecraft."
    *   **Resposta do LLM (Tentativa 1):** (Parágrafo genérico, sem o vocabulário ou a atmosfera cósmica de Lovecraft)
    *   **Prompt (Reflexão):** "O parágrafo gerado é muito direto e carece da adjetivação densa e do tom de horror cósmico de Lovecraft. Reflita sobre o uso de palavras como 'indescritível', 'eldritch', 'abissal' e 'insanidade'. Reescreva o parágrafo, focando em evocar uma sensação de pavor e mistério inominável."

5.  **Exemplo 5: Otimização de Prompt (Meta-Reflexão)**
    *   **Prompt (Usuário):** "Crie um prompt de engenharia para um LLM que o instrua a atuar como um especialista em segurança cibernética e analisar um trecho de código em busca de vulnerabilidades XSS."
    *   **Resposta do LLM (Tentativa 1):** (Prompt simples, sem definir persona, formato de saída ou etapas de análise)
    *   **Prompt (Reflexão):** "O prompt de engenharia criado é ineficaz porque não estabelece uma persona clara nem um formato de saída estruturado. Reflita sobre as melhores práticas de engenharia de prompt (e.g., CoT, persona, formato JSON) e reescreva o prompt para ser mais robusto. O novo prompt deve incluir: 1) A persona de 'Especialista em Segurança Nível 5', 2) A instrução para usar o raciocínio 'Chain-of-Thought' e 3) A exigência de saída em formato Markdown com uma tabela de vulnerabilidades."

6.  **Exemplo 6: Refinando um Argumento (Debate)**
    *   **Prompt (Usuário):** "Escreva um argumento de 200 palavras a favor da implementação de uma semana de trabalho de 4 dias."
    *   **Resposta do LLM (Tentativa 1):** (Argumento focado apenas em bem-estar, ignorando produtividade e custos)
    *   **Prompt (Reflexão):** "O argumento anterior é fraco porque não aborda as preocupações econômicas e de produtividade. Reflita sobre como integrar dados sobre o aumento da eficiência e a redução de custos operacionais (e.g., energia, escritório) para fortalecer a tese. Reescreva o argumento para ser mais equilibrado e persuasivo para um público corporativo."

7.  **Exemplo 7: Correção de Dados (Factual)**
    *   **Prompt (Usuário):** "Liste os 5 maiores rios do mundo em ordem de comprimento."
    *   **Resposta do LLM (Tentativa 1):** (Lista incorreta, colocando o Rio Nilo em primeiro lugar, quando o Amazonas é geralmente aceito como o mais longo)
    *   **Prompt (Reflexão):** "A lista anterior está desatualizada ou incorreta. Fontes modernas de pesquisa geográfica indicam que o Rio Amazonas é o mais longo do mundo, superando o Nilo. Revise a lista com base nas medições mais aceitas e corrija a ordem. A reflexão deve ser: 'A informação factual sobre o Rio Amazonas e Nilo foi revisada. O Amazonas é o mais longo. A lista será corrigida.'"
```

## Best Practices
*   **Instrução Clara para a Reflexão:** O prompt de reflexão deve ser específico, solicitando ao modelo que identifique a causa raiz da falha, sugira uma correção e formule uma nova estratégia.
*   **Memória Dinâmica:** Mantenha um histórico conciso das tentativas e das reflexões anteriores no contexto do prompt para a nova tentativa. A reflexão deve ser a parte mais relevante da memória.
*   **Iteração Controlada:** Limite o número de iterações para evitar loops infinitos ou consumo excessivo de recursos. Três a cinco iterações costumam ser suficientes.
*   **Combinação com ReAct:** Use o Reflexion para aprimorar o componente de "Raciocínio" (Reasoning) do ReAct, permitindo que o agente aprenda com suas interações com o ambiente.
*   **Foco na Causa do Erro:** Oriente o modelo a refletir sobre o *porquê* a tentativa falhou, e não apenas *o que* falhou.

## Use Cases
*   **Geração de Código:** Agentes que tentam escrever código, recebem erros de compilação ou falhas em testes unitários, refletem sobre o erro e corrigem o código.
*   **Resolução de Problemas Complexos (e.g., Matemática, Lógica):** O agente tenta resolver um problema, avalia a resposta, identifica um erro de raciocínio e refina a cadeia de pensamento.
*   **Navegação e Interação em Ambientes Virtuais:** Agentes que interagem com APIs ou ambientes de jogos, recebem *feedback* do ambiente (e.g., "ação inválida"), refletem sobre a estratégia e ajustam o plano de ação.
*   **Criação de Conteúdo Iterativo:** Refinar um artigo, roteiro ou peça criativa com base em critérios de avaliação internos ou externos.

## Pitfalls
*   **Reflexões Superficiais:** O modelo pode gerar reflexões genéricas ou superficiais que não levam a melhorias reais na próxima tentativa.
*   **Acúmulo de Contexto (Context Window Bloat):** O histórico de tentativas e reflexões pode rapidamente exceder o limite de contexto do LLM, exigindo estratégias de sumarização ou poda de memória.
*   **Loops de Reflexão:** Em casos raros, o agente pode entrar em um ciclo onde a reflexão não consegue quebrar o padrão de erro, levando a tentativas repetitivas e infrutíferas.
*   **Custo Computacional:** O processo iterativo e a necessidade de múltiplas chamadas ao LLM (tentativa + reflexão + nova tentativa) aumentam significativamente o custo e o tempo de resposta.

## URL
[https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
