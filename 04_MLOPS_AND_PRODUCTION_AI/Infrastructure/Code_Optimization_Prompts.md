# Code Optimization Prompts

## Description
Prompts de Otimização de Código são instruções específicas fornecidas a Large Language Models (LLMs) para solicitar a análise, refatoração e melhoria de código-fonte. O objetivo é aprimorar o desempenho, a eficiência, a legibilidade, a segurança e a adesão a melhores práticas de programação. A técnica frequentemente utiliza abordagens como **Chain-of-Thought (CoT)**, solicitando ao LLM que estruture seu raciocínio (identificar gargalos, propor otimizações e explicar o impacto) antes de fornecer o código final. Esta técnica posiciona a IA como uma aliada poderosa para escalar a produtividade do desenvolvedor, liberando tempo para focar em arquitetura e lógica de negócio complexa.

## Examples
```
1. **Otimização de Consultas SQL:** "Assumindo o papel de um **DBA Sênior**, analise a seguinte query SQL para um banco de dados **PostgreSQL** de uma aplicação de marketplace. A query está causando lentidão em relatórios diários de vendas. Identifique gargalos e reescreva a query de forma otimizada. Sugira **índices adicionais** e explique o impacto de cada otimização no plano de execução. [INSERIR QUERY AQUI]"
2. **Refatoração para Desempenho:** "Refatore o seguinte bloco de código Python. O objetivo é melhorar a **eficiência de tempo de execução (time complexity)**, especialmente para grandes volumes de dados. Explique a complexidade O(n) do código original e do código refatorado."
3. **Otimização de Uso de Memória:** "Analise a função JavaScript abaixo, que processa um grande array de objetos. Identifique e elimine quaisquer **vazamentos de memória (memory leaks)** ou padrões que causem alto consumo de memória. Otimize a função para um uso de memória mais eficiente."
4. **Revisão de Segurança e Performance:** "Atue como um **Engenheiro de Segurança e Performance**. Revise o trecho de código Java Spring Boot. Identifique vulnerabilidades de segurança (ex: injeção de SQL, XSS) e gargalos de performance. Forneça o código corrigido e uma explicação detalhada das alterações."
5. **Otimização de Algoritmo:** "O algoritmo a seguir implementa a busca por [NOME DO ALGORITMO]. Ele está lento. Reescreva-o usando uma abordagem mais eficiente, como [SUGESTÃO DE ALGORITMO MAIS EFICIENTE, ex: programação dinâmica]. Compare o desempenho teórico das duas implementações."
6. **Debugging e Análise de Logs:** "Analise o seguinte trecho de log de erro de uma aplicação Java Spring Boot. Identifique a **causa raiz do problema**, sugira uma **solução de código** para resolvê-la e proponha um **teste unitário** para evitar a regressão. [INSERIR LOG E CÓDIGO AQUI]"
7. **Otimização de Expressões Regulares:** "A expressão regular `[INSERIR REGEX AQUI]` está demorando muito para processar strings longas, causando problemas de **backtracking excessivo**. Otimize a expressão regular para ser mais eficiente e evite o 'catastrophic backtracking'. Explique a otimização."
```

## Best Practices
**Definir o Papel (Role Prompting):** Atribuir um papel específico (ex: DBA Sênior, Engenheiro de Performance) ao LLM aumenta a qualidade e o foco da resposta.
**Especificar o Contexto:** Incluir a linguagem, framework, versão e o ambiente de execução (ex: PostgreSQL, Node.js, Spring Boot) é crucial para uma análise precisa.
**Definir o Objetivo de Otimização:** Ser explícito sobre o que precisa ser otimizado (tempo de execução, uso de memória, segurança, legibilidade).
**Solicitar o Raciocínio (Chain-of-Thought - CoT):** Pedir ao LLM para "explicar o impacto", "identificar gargalos" ou "comparar complexidades" garante uma resposta mais robusta e verificável.
**Fornecer o Código/Log Completo:** O LLM precisa do contexto completo para uma análise precisa.
**Validação e Teste:** Sempre validar e testar o código otimizado, pois a IA é uma ferramenta, não um substituto para o senso crítico.

## Use Cases
**Refatoração de Código Legado:** Melhorar a performance e a legibilidade de bases de código antigas.
**Otimização de Consultas de Banco de Dados:** Reduzir o tempo de resposta de queries lentas.
**Identificação de Vulnerabilidades de Segurança:** Revisão de código para falhas de segurança e aderência a padrões.
**Redução de Custos de Infraestrutura:** Otimizar o código para consumir menos CPU/Memória, reduzindo custos de cloud.
**Geração de Testes de Regressão:** Criar testes unitários para garantir que a otimização não introduza novos bugs.
**Melhoria da Complexidade Algorítmica:** Transformar algoritmos ineficientes (ex: O(n²)) em soluções mais rápidas (ex: O(n log n)).

## Pitfalls
**Confiança Cega:** Aceitar a otimização da IA sem validação manual ou testes de benchmark.
**Falta de Contexto:** Não fornecer o código completo ou o ambiente de execução, levando a otimizações incorretas ou incompletas.
**Otimização Prematura:** Otimizar código que não é o verdadeiro gargalo do sistema (o problema real pode estar em outro lugar).
**Introdução de Bugs:** O código otimizado pode ser mais complexo e introduzir novos erros lógicos.
**Perda de Legibilidade:** Otimizações extremas podem tornar o código menos legível e mais difícil de manter.

## URL
[https://www.programaria.org/como-turbinar-prompts-para-seu-codigo-com-ia-generativa/](https://www.programaria.org/como-turbinar-prompts-para-seu-codigo-com-ia-generativa/)
