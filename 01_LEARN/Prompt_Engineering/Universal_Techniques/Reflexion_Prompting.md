# Reflexion Prompting

## Description

**Reflexion Prompting** é um *framework* inovador que aprimora agentes de linguagem (LLMs) através de **feedback linguístico** e **memória episódica**, em vez de ajustes de peso do modelo (fine-tuning). O agente Reflexion executa uma tarefa e, se falhar, ele *reflete* verbalmente sobre o resultado, diagnosticando a causa da falha e formulando um novo plano de ação. Essa reflexão é então armazenada em uma memória episódica e utilizada como contexto para a próxima tentativa de resolver a mesma tarefa.

O processo é iterativo e auto-corretivo, permitindo que o agente aprenda com a **tentativa e erro** de forma eficiente. A reflexão atua como um mecanismo de **reforço verbal**, onde o agente se auto-corrige e melhora sua estratégia de raciocínio e tomada de decisão ao longo de múltiplas tentativas.

A técnica é agnóstica ao modelo e pode ser aplicada a diversos tipos de tarefas, como raciocínio, tomada de decisão sequencial e programação. A chave é a capacidade do agente de transformar o feedback do ambiente (sucesso/falha) em uma **reflexão em linguagem natural** que guia seu comportamento futuro.

## Statistics

O artigo original de Shinn et al. (2023) demonstrou ganhos significativos de desempenho:
- **Programação (HumanEval):** Reflexion alcançou **91% de acurácia Pass@1**, superando o estado da arte anterior do GPT-4 (80%).
- **Raciocínio (HotPotQA):** Em tarefas de raciocínio de múltiplas etapas, Reflexion superou o agente ReAct em **11%** na taxa de sucesso.
- **Tomada de Decisão Sequencial (AlfWorld):** Reflexion melhorou a taxa de sucesso do agente ReAct em **22%** em tarefas complexas de navegação e manipulação de objetos.
- **Citações:** O artigo "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al., 2023) é altamente influente, com **mais de 2500 citações** (em Nov 2025), indicando sua relevância e impacto na pesquisa de agentes de linguagem.

## Features

- **Reforço Verbal:** Utiliza feedback linguístico (a reflexão) para aprimorar o agente, sem a necessidade de retreinamento do modelo.
- **Memória Episódica:** Armazena as reflexões geradas em uma memória que é consultada em tentativas subsequentes.
- **Auto-Correção:** O agente diagnostica suas próprias falhas e formula um plano de ação para mitigar erros futuros.
- **Agnóstico ao Modelo:** Pode ser aplicado a qualquer LLM que suporte o encadeamento de pensamentos (como ReAct ou CoT).
- **Melhoria Iterativa:** O desempenho do agente melhora progressivamente a cada tentativa e reflexão.

## Use Cases

- **Agentes de Programação:** Melhorar a capacidade de LLMs de gerar código funcional e corrigir erros de forma autônoma (ex: HumanEval, LeetCode).
- **Raciocínio de Múltiplas Etapas:** Aprimorar a precisão em tarefas de perguntas e respostas que exigem múltiplas etapas de raciocínio e busca de informações (ex: HotPotQA).
- **Tomada de Decisão em Ambientes Virtuais:** Guiar agentes em ambientes de simulação (ex: AlfWorld, WebShop) para completar tarefas sequenciais complexas, como navegação e interação com objetos.
- **Automação de Tarefas:** Criação de agentes mais robustos e autônomos para tarefas do mundo real que envolvem tentativa e erro e feedback externo.

## Integration

**Exemplo de Prompt para Geração de Reflexão (Adaptado do paper):**

```
Você é um agente de raciocínio avançado que pode melhorar com base na auto-reflexão. Você receberá uma tentativa de raciocínio anterior na qual teve acesso a um ambiente de API Docstore e uma pergunta para responder. Você não conseguiu responder à pergunta, seja porque adivinhou a resposta errada com Finish[<resposta>], ou porque esgotou o número definido de etapas de raciocínio. Em poucas frases, diagnostique uma possível razão para a falha e crie um novo plano conciso e de alto nível que vise mitigar a mesma falha. Use frases completas.

Tentativa anterior:
Pergunta: {question}
{scratchpad}

Reflexão:
```

**Melhores Práticas:**
1.  **Diagnóstico Claro:** A reflexão deve diagnosticar claramente a causa da falha (ex: "A busca inicial foi muito restrita" ou "O agente clicou no item errado devido ao preço").
2.  **Plano de Ação Concreto:** O plano deve ser conciso e fornecer uma estratégia de alto nível para a próxima tentativa (ex: "Na próxima vez, farei uma busca mais ampla" ou "Vou verificar o preço antes de clicar em 'Comprar'").
3.  **Integração com ReAct/CoT:** Reflexion é frequentemente usado em conjunto com técnicas como ReAct (Reasoning and Acting) ou CoT (Chain-of-Thought), onde a reflexão é injetada no prompt do agente para guiar o próximo ciclo de raciocínio.
4.  **Memória Persistente:** A reflexão deve ser armazenada e persistir para ser usada em todas as tentativas subsequentes da mesma tarefa. O prompt do agente deve incluir um cabeçalho como: `Você tentou responder à seguinte pergunta antes e falhou. A(s) seguinte(s) reflexão(ões) fornece(m) um plano para evitar falhar ao responder à pergunta da mesma forma que você fez anteriormente. Use-as para melhorar sua estratégia...`

## URL

https://arxiv.org/abs/2303.11366