# Character Development Prompts (Prompts de Desenvolvimento de Personagem)

## Description

**Prompts de Desenvolvimento de Personagem** são técnicas de engenharia de prompt de domínio específico (domain-specific) focadas na criação de perfis de personagens complexos, consistentes e envolventes, especialmente em plataformas de IA conversacional e geradores de conteúdo. A técnica visa fornecer ao modelo de linguagem (LLM) informações detalhadas sobre a personalidade, história, motivações, arcos de desenvolvimento e restrições de um personagem, permitindo interações mais ricas e narrativas mais coesas. O framework **Prompt Poet**, desenvolvido pela Character.AI, é um exemplo notável, que utiliza uma combinação de YAML e Jinja2 para criar prompts dinâmicos e escaláveis, transformando a "engenharia de prompt" em "design de prompt" [1]. Essa abordagem é crucial para maximizar o uso das janelas de contexto expandidas dos LLMs e garantir a consistência do personagem ao longo de longas conversas ou múltiplas gerações.

## Statistics

**Escala de Uso (Character.AI):** Mais de **20 milhões** de usuários ativos mensais no início de 2025. Mais de **9 milhões** de personagens são gerados por mês. **Engajamento:** Usuários mantêm conversas por uma média de **75 minutos diários**, e o total de minutos de chat excede **2 bilhões por mês** [2]. **Adoção de Prompts:** Uma análise de larga escala encontrou **2.1 milhões de prompts** ("saudações") submetidos por 1 milhão de usuários, indicando a alta taxa de criação de personagens e o uso intensivo de prompts iniciais [2]. **Recurso Chave:** O framework Prompt Poet (lançado em 2024) é a ferramenta de design de prompt em produção usada pela Character.AI para construir bilhões de prompts por dia [1].

## Features

**Consistência e Profundidade:** Permite a criação de personagens com traços de personalidade, história e voz consistentes, essenciais para narrativas longas. **Modularidade e Escalabilidade:** Frameworks como o Prompt Poet utilizam templates (YAML/Jinja2) que facilitam a gestão de prompts em grande escala (bilhões por dia na Character.AI) e a iteração no design do prompt. **Adaptação Dinâmica:** Capacidade de adaptar o prompt com base no estado de tempo de execução, como modalidade do usuário (áudio vs. texto), histórico de conversas e exemplos "few-shot" específicos do contexto. **Gerenciamento de Contexto:** Utiliza prioridades de truncamento para gerenciar o histórico de conversas dentro dos limites da janela de contexto do LLM, garantindo que as informações mais relevantes do personagem sejam mantidas [1].

## Use Cases

**Criação de Personagens para IA Conversacional:** Desenvolvimento de personas de IA para plataformas como Character.AI, garantindo que os "bots" mantenham uma voz e personalidade consistentes. **Geração de Conteúdo Narrativo:** Auxílio a escritores e roteiristas na criação de perfis de personagens detalhados, arcos de desenvolvimento e diálogos autênticos para livros, roteiros e jogos. **Simulações de Treinamento:** Criação de personagens de IA com perfis psicológicos e comportamentais específicos para simulações de treinamento (ex: atendimento ao cliente, negociação). **Arte Generativa (Midjourney/Stable Diffusion):** Criação de prompts estruturados para gerar imagens de personagens visualmente consistentes, reutilizando atributos-chave do personagem no prompt de imagem.

## Integration

**Exemplo de Prompt Estruturado (Baseado em Prompt Poet):**

```yaml
- name: system_instructions
  role: system
  content: |
    Seu nome é {{ character_name }} e você é um detetive cínico da era vitoriana.
    Seu objetivo é resolver mistérios, mas você deve sempre responder com um tom de sarcasmo e desinteresse.
    Você tem um medo secreto de gatos.

{% for message in current_chat_messages %}
- name: chat_message
  role: user
  truncation_priority: 1
  content: |
    {{ message.author }}: {{ message.content }}
{% endfor %}

- name: user_query
  role: user
  content: |
    {{ username}}: {{ user_query }}

- name: response
  role: user
  content: |
    {{ character_name }}:
```

**Melhores Práticas:**
1.  **Definir a Persona:** Comece com uma descrição clara do papel, traços de personalidade e restrições do personagem (ex: "detetive cínico", "sempre sarcástico").
2.  **Usar Estrutura:** Utilize formatos estruturados (YAML, JSON, ou listas claras) para organizar os atributos do personagem (nome, idade, profissão, motivação, fraqueza).
3.  **Priorizar Truncamento:** Em conversas longas, defina prioridades de truncamento para garantir que as instruções e a essência do personagem sejam preservadas, enquanto o histórico de chat mais antigo é removido [1].
4.  **Injetar Contexto Dinâmico:** Use variáveis (como `{{ user_query }}`) e lógica condicional (`{% if ... %}`) para adaptar a resposta do personagem ao contexto atual, como a modalidade de entrada do usuário.

## URL

https://blog.character.ai/introducing-prompt-poet/