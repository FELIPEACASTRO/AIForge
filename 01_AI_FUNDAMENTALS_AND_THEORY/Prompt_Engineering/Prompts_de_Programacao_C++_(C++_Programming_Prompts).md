# Prompts de Programação C++ (C++ Programming Prompts)

## Description
**Prompts de Programação C++** referem-se a instruções estruturadas e detalhadas fornecidas a Modelos de Linguagem Grande (LLMs) para gerar, analisar, refatorar, documentar ou depurar código C++. Devido à complexidade, sintaxe rigorosa e foco em desempenho e gerenciamento de memória do C++, a engenharia de prompt para esta linguagem exige um alto grau de precisão e contexto. A técnica eficaz se baseia em quatro pilares principais: **Persona** (definir o papel da IA), **Contexto** (fornecer detalhes técnicos como versão do padrão C++, bibliotecas e restrições de sistema), **Tarefa** (o objetivo claro) e **Formato** (a estrutura de saída desejada). A pesquisa recente (2023-2025) enfatiza a abordagem **sinérgica** de prompts, combinando técnicas como *Chain-of-Thought* (CoT) e *Few-Shot* para melhorar a precisão e a segurança do código gerado, especialmente em domínios críticos como sistemas embarcados e desenvolvimento de jogos. O objetivo é mitigar a tendência dos LLMs de produzir código funcional, mas inseguro ou ineficiente, que é um risco particular no desenvolvimento em C++.

## Examples
```
**1. Geração de Código Otimizado para Sistema Embarcado**
```
Persona: Você é um engenheiro de software C++ sênior, especialista em sistemas embarcados com restrições de memória.
Contexto: Estou usando C++17 em um microcontrolador ARM Cortex-M4. A função deve ser otimizada para velocidade e ter pegada de memória mínima.
Tarefa: Escreva uma função C++ que implemente um filtro de média móvel (Moving Average Filter) para um array de 100 inteiros de 16 bits. Use `std::array` e evite alocação dinâmica de memória.
Formato: Forneça apenas o código-fonte C++ completo, incluindo a função e um pequeno `main` de exemplo.
```

**2. Refatoração e Modernização de Código Legado**
```
Persona: Você é um arquiteto de software C++ com foco em modernização de código.
Contexto: O trecho de código C++ a seguir usa ponteiros brutos e alocação manual. [Incluir trecho de código legado]
Tarefa: Refatore o código para usar ponteiros inteligentes (smart pointers) do C++ moderno (C++20), como `std::unique_ptr` e `std::shared_ptr`, para garantir o gerenciamento seguro de memória. Mantenha a funcionalidade original.
Formato: Apresente o código refatorado e uma breve explicação das mudanças em uma lista numerada.
```

**3. Análise de Segurança e Revisão de Código**
```
Persona: Você é um especialista em segurança de software, focado em vulnerabilidades de C/C++.
Contexto: Analise o seguinte código C++ que lida com entrada de usuário. [Incluir código C++]
Tarefa: Identifique potenciais vulnerabilidades de segurança, como buffer overflows, race conditions ou memory leaks. Sugira correções específicas e explique o risco de cada vulnerabilidade.
Formato: Use uma tabela Markdown com as colunas: 'Vulnerabilidade', 'Severidade (Alta/Média/Baixa)', 'Explicação', 'Correção Sugerida'.
```

**4. Geração de Testes Unitários com Framework Específico**
```
Persona: Você é um engenheiro de QA com experiência em testes unitários C++.
Contexto: A classe C++ `Calculadora` possui métodos para adição, subtração e divisão. [Incluir cabeçalho da classe `Calculadora.h`]
Tarefa: Gere um conjunto abrangente de testes unitários para a classe `Calculadora` usando o framework **Google Test**. Inclua testes de caso de borda, como divisão por zero.
Formato: Forneça o arquivo `.cpp` de teste completo, pronto para compilação.
```

**5. Explicação de Conceitos Avançados de C++**
```
Persona: Você é um instrutor didático de C++ com foco em clareza e exemplos práticos.
Contexto: Meu nível é intermediário e estou aprendendo C++20.
Tarefa: Explique o conceito de **Coroutines** em C++20. Inclua um pequeno exemplo de código que demonstre o uso de uma coroutine simples para uma operação assíncrona.
Formato: Use parágrafos claros para a explicação e um bloco de código C++ bem comentado para o exemplo.
```

**6. Geração de Código com Restrições de Design Pattern**
```
Persona: Você é um desenvolvedor C++ experiente em Design Patterns.
Contexto: Estou implementando um sistema de log.
Tarefa: Escreva a implementação C++ (arquivo .h e .cpp) de um **Design Pattern Singleton** thread-safe para uma classe de Log. Use a inicialização *Meyers' Singleton* (Magic Static) para garantir a segurança de thread e a inicialização preguiçosa.
Formato: Forneça os dois arquivos (`Log.h` e `Log.cpp`) separadamente.
```
```

## Best Practices
**Definir a Persona (Role-Playing):** Sempre comece o prompt definindo o papel da IA (ex: "Você é um engenheiro de software C++ sênior, especialista em sistemas embarcados e otimização de desempenho"). Isso alinha a resposta ao nível de conhecimento e estilo desejado.
**Fornecer Contexto Detalhado:** Inclua o máximo de contexto possível, como a versão do C++ (C++17, C++20, C++23), as bibliotecas a serem usadas (Boost, Qt, STL, etc.), o ambiente de destino (Linux, Windows, microcontrolador) e as restrições de desempenho ou memória.
**Especificar o Formato de Saída:** Peça explicitamente o formato desejado (ex: "Forneça apenas o código, sem explicações", "Use comentários Doxygen para documentação", "Apresente a análise em uma tabela Markdown").
**Dividir Tarefas Complexas:** Em vez de pedir para "Criar um servidor web C++", divida em etapas: "1. Escreva a classe de soquete TCP", "2. Implemente o loop de eventos", "3. Adicione o tratamento de erros".
**Incluir Exemplos (Few-Shot):** Para garantir que o código gerado siga um estilo ou padrão específico da sua base de código, inclua um pequeno trecho de código C++ existente como exemplo.
**Foco em Segurança e Robustez:** Peça explicitamente por código seguro e robusto, solicitando a verificação de *memory leaks*, *buffer overflows* e o uso de práticas modernas (ex: `std::unique_ptr`, `std::span`).

## Use Cases
**Geração de Código Otimizado:** Criar funções e classes C++ que atendam a requisitos estritos de desempenho e uso de recursos, comuns em *High-Frequency Trading* ou *Game Engines*.
**Modernização de Código Legado:** Refatorar bases de código antigas (C++98/03) para padrões modernos (C++17/20/23), substituindo ponteiros brutos por *smart pointers* e usando recursos como *Concepts* e *Modules*.
**Análise de Código para Segurança:** Identificar e corrigir vulnerabilidades de segurança específicas do C/C++, como *buffer overflows* e *integer overflows*, antes da revisão humana.
**Geração de Testes Unitários:** Criar rapidamente conjuntos de testes abrangentes usando frameworks como Google Test ou Catch2, incluindo casos de borda e testes de estresse.
**Documentação Técnica:** Gerar documentação Doxygen ou comentários de código claros e consistentes para grandes projetos C++, garantindo a manutenibilidade a longo prazo.
**Explicação de Conceitos Avançados:** Usar a IA como um tutor para explicar recursos complexos do C++ (ex: *metaprogramação de templates*, *variadic templates*, *Coroutines*) com exemplos práticos e funcionais.

## Pitfalls
**Prompts Vagos ou Genéricos:** Pedir "Escreva um código C++" sem especificar a versão do padrão (C++11 vs C++20), o ambiente ou as bibliotecas. Isso resulta em código genérico, potencialmente obsoleto ou incompatível.
**Ignorar o Gerenciamento de Memória:** Não mencionar ponteiros inteligentes ou alocação de memória. O LLM pode gerar código com `new` e `delete` brutos, introduzindo *memory leaks* ou falhas de segmentação.
**Falta de Contexto de Desempenho:** Não especificar a necessidade de otimização. O LLM pode usar estruturas de dados ou algoritmos de alto nível que são lentos ou consomem muita memória, o que é crítico em C++.
**Confiar Cegamente na Segurança:** O código gerado pode parecer correto, mas conter vulnerabilidades de segurança (ex: uso inseguro de `strcpy` ou `scanf`). É crucial solicitar explicitamente a verificação de segurança.
**Sobrecarga de Tarefas:** Tentar refatorar, documentar e otimizar um grande bloco de código em um único prompt. Isso confunde o modelo e reduz a qualidade da saída. Divida em prompts menores e sequenciais.

## URL
[https://blogs.sw.siemens.com/thought-leadership/prompt-engineering-part-2-best-practices-for-software-developers-in-digital-industries/](https://blogs.sw.siemens.com/thought-leadership/prompt-engineering-part-2-best-practices-for-software-developers-in-digital-industries/)
