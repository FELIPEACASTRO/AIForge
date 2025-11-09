# Chain-of-Thought (CoT) Prompting

## Description

O **Chain-of-Thought (CoT) Prompting** é uma técnica de engenharia de prompt que capacita Modelos de Linguagem Grande (LLMs) a realizar raciocínio complexo, guiando-os a gerar uma sequência de passos de raciocínio intermediários antes de fornecer a resposta final. Essa abordagem simula o processo de pensamento humano, transformando problemas complexos em etapas gerenciáveis. O CoT é particularmente eficaz para tarefas que exigem raciocínio multi-passos, como matemática, lógica e senso comum. A técnica original (Few-Shot CoT) requer exemplos de entrada/saída com a "cadeia de pensamento" explícita, mas variações como o **Zero-Shot CoT** (adicionando "Let's think step by step" ou "Pense passo a passo") tornaram-no acessível sem a necessidade de exemplos. Desenvolvimentos recentes (2025) incluem o **Layered CoT** (raciocínio em múltiplas camadas com revisão), **Trace-of-Thought** (para modelos menores) e **LongRePS** (para raciocínio em contextos longos). No entanto, o CoT pode aumentar a latência e o custo devido ao maior número de *tokens* gerados, e sua eficácia é mais pronunciada em modelos com mais de 100 bilhões de parâmetros.

## Statistics

- **Melhoria de Desempenho (PaLM 540B):**
    - **GSM8K (Matemática):** Acurácia melhorou de 55% para **74%** (+19%).
    - **SVAMP (Matemática):** Acurácia melhorou de 57% para **81%** (+24%).
    - **Raciocínio Simbólico:** Acurácia melhorou de ~60% para **~95%** (+35%).
- **Limitação de Escala:** A técnica CoT só produz ganhos de desempenho significativos quando usada com modelos de **~100 Bilhões de parâmetros** ou mais. Modelos menores podem ter a acurácia reduzida.
- **Citações:** O artigo original, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022), é um dos mais citados na área de LLMs, com mais de **21.800 citações** (até 2025).
- **Custo/Latência:** O CoT aumenta o número de *tokens* gerados, resultando em maior latência e custo de inferência em produção. Em modelos de raciocínio mais recentes, os ganhos de acurácia podem ser marginais (2-3%), tornando a compensação custo/benefício um fator crítico.

## Features

- **Raciocínio Multi-Passos:** Permite que LLMs decomponham problemas complexos em etapas lógicas sequenciais.
- **Transparência e Auditabilidade:** A cadeia de pensamento gerada oferece visibilidade sobre como o modelo chegou à sua resposta, aumentando a confiança e permitindo a depuração.
- **Melhoria de Desempenho:** Aumenta significativamente a precisão em tarefas de raciocínio, especialmente em modelos grandes.
- **Variações Avançadas:** Inclui Zero-Shot CoT (sem exemplos), Automatic CoT (Auto-CoT, para amostragem de demonstrações) e técnicas mais recentes como Layered CoT, Trace-of-Thought e LongRePS.
- **Emergência de Capacidade:** É uma capacidade emergente que se manifesta em modelos de grande escala (tipicamente >100B de parâmetros).

## Use Cases

- **Resolução de Problemas Matemáticos:** Tarefas de raciocínio aritmético e algébrico complexas (e.g., benchmarks GSM8K e SVAMP).
- **Raciocínio de Senso Comum:** Resolução de questões que exigem inferência e lógica multi-passos (e.g., benchmark CSQA).
- **Raciocínio Simbólico:** Tarefas que envolvem manipulação de símbolos e regras lógicas.
- **Planejamento e Tomada de Decisão:** Simulação de etapas de planejamento para agentes de IA e sistemas de tomada de decisão.
- **Serviço de Atendimento ao Cliente (Gen-AI Backed):** Chatbots que precisam analisar a intenção do usuário, consultar múltiplas fontes de dados e formular uma resposta estruturada.
- **Aplicações de Alto Risco (Layered CoT):** Uso em saúde ou finanças, onde a revisão e o ajuste do raciocínio em múltiplas camadas são cruciais.

## Integration

O CoT pode ser implementado de duas formas principais:

**1. Few-Shot CoT (CoT com Exemplos):**
Inclua no prompt de entrada um ou mais exemplos de pares pergunta/resposta onde a resposta contém a "cadeia de pensamento" explícita.

*Exemplo de Prompt (Matemática):*
```
Q: Roger tem 5 bolas de tênis. Ele compra mais 2 latas de bolas de tênis. Cada lata tem 3 bolas de tênis. Quantas bolas de tênis ele tem agora?
A: Roger começou com 5 bolas. Ele comprou 2 latas de 3 bolas de tênis cada, o que é 6 bolas de tênis. 5 + 6 = 11. A resposta é 11.

Q: A cafeteria tinha 23 maçãs. Se eles usaram 20 para fazer o almoço e compraram mais 6, quantas maçãs eles têm?
A: A cafeteria tinha 23 maçãs originalmente. Eles usaram 20 para o almoço, então ficaram com 23 - 20 = 3. Eles compraram mais 6 maçãs, então eles têm 3 + 6 = 9. A resposta é 9.
```

**2. Zero-Shot CoT (CoT de Zero-Exemplo):**
Adicione uma frase simples ao final do prompt para instruir o modelo a raciocinar.

*Exemplo de Prompt (Senso Comum/Lógica):*
```
Q: Fui ao mercado e comprei 10 maçãs. Dei 2 maçãs ao vizinho e 2 ao técnico. Depois comprei mais 5 maçãs e comi 1. Com quantas maçãs eu fiquei?

Pense passo a passo.
```
*Melhores Práticas:*
- **Modelos Grandes:** Use CoT principalmente com LLMs de grande escala (tipicamente >100B de parâmetros), pois modelos menores podem gerar raciocínios ilógicos.
- **Tarefas Complexas:** Reserve o CoT para tarefas que exigem raciocínio complexo (matemática, lógica, planejamento) e evite-o para tarefas simples, onde adiciona latência e custo desnecessários.
- **Variações:** Experimente o Zero-Shot CoT primeiro pela sua simplicidade. Para tarefas críticas, considere o uso de técnicas como **Self-Consistency** (gerar múltiplas cadeias de pensamento e escolher a resposta mais comum) para aumentar a robustez.

## URL

https://arxiv.org/abs/2201.11903