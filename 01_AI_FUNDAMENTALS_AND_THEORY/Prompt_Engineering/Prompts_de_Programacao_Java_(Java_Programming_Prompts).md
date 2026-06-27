# Prompts de Programação Java (Java Programming Prompts)

## Description
**Prompts de Programação Java** referem-se à arte e ciência de criar instruções otimizadas para modelos de linguagem grandes (LLMs) com o objetivo de gerar, analisar, refatorar, documentar ou depurar código no ecossistema Java. Esta técnica é crucial para desenvolvedores que buscam alavancar ferramentas de IA (como GitHub Copilot, Amazon CodeWhisperer ou Spring AI) para aumentar a produtividade e a qualidade do código [1] [2].

A eficácia de um prompt de programação Java depende da clareza, do contexto fornecido e da especificação do formato de saída desejado. O uso de técnicas avançadas como *System Prompting* (definição de função), *Few-Shot Prompting* (exemplos de formato) e *Chain-of-Thought* (raciocínio passo a passo) é fundamental para superar as limitações dos LLMs, como a alucinação de APIs ou a geração de código inseguro ou ineficiente [3] [4].

O ecossistema Java, com sua sintaxe verbosa e forte tipagem, beneficia-se enormemente de prompts bem elaborados que transformam tarefas repetitivas em código funcional e testável. A tendência é que o *Prompt Engineering* se torne uma habilidade essencial para o desenvolvedor Java moderno, especialmente com a integração de LLMs em frameworks como o Spring AI [1].

## Examples
```
**1. Geração de Código com Especificação de Framework e Versão**
```
// Prompt:
"Você é um desenvolvedor Java sênior. Gere uma classe de serviço Spring Boot 3.2 que implemente a interface 'UserService'. O método 'createUser(User user)' deve usar 'JdbcTemplate' para inserir um novo usuário no banco de dados 'users'. Use Java 21 e garanta que o código siga as melhores práticas de injeção de dependência."
```

**2. Refatoração e Otimização de Código**
```
// Prompt:
"Refatore o seguinte trecho de código Java para usar a API Stream e melhorar a legibilidade. O objetivo é filtrar uma lista de objetos 'Produto' onde 'estoque > 0' e mapear para uma lista de seus nomes. Adicione tratamento de exceção para 'NullPointerException' de forma concisa.

[CÓDIGO A SER REFATORADO]"
```

**3. Geração de Testes Unitários com Casos de Borda**
```
// Prompt:
"Para a classe Java 'Calculadora' fornecida abaixo, escreva testes unitários completos usando JUnit 5. Inclua testes para os métodos 'somar', 'subtrair' e 'dividir'. Garanta que haja um teste específico para o caso de divisão por zero, esperando a exceção 'ArithmeticException'.

[CÓDIGO DA CLASSE CALCULADORA]"
```

**4. Documentação e Explicação de Código Legado**
```
// Prompt:
"Analise o código Java legado abaixo. Explique, em português, a funcionalidade de cada método e a arquitetura geral da classe. Sugira melhorias de design e documente o código com anotações Javadoc completas.

[CÓDIGO LEGADO]"
```

**5. Mapeamento de Resposta Estruturada (JSON/POJO)**
```
// Prompt (com System Prompt implícito):
"Classifique o sentimento do seguinte comentário de cliente sobre um aplicativo Java em POSITIVO, NEUTRO ou NEGATIVO. Retorne a resposta estritamente no formato JSON, mapeável para um POJO Java com os campos 'sentimento' (String) e 'confianca' (Double).

Comentário: 'A nova versão do app está incrivelmente rápida, mas a interface ficou confusa.'"
```

**6. Correção de Bug e Análise de Stack Trace**
```
// Prompt:
"Analise o seguinte Stack Trace e o trecho de código Java. Identifique a causa raiz do 'NullPointerException' e forneça o código corrigido. Explique a correção em uma frase.

[STACK TRACE]
[TRECHO DE CÓDIGO]"
```
```

## Best Practices
**1. Seja Específico e Contextualizado (System Prompting):** Defina claramente o papel da IA (ex: "Você é um desenvolvedor Java sênior com 10 anos de experiência em Spring Boot") e forneça o contexto relevante (versão do Java, frameworks, trechos de código existentes) [1] [2].
**2. Use o Formato Few-Shot para Estrutura:** Para tarefas que exigem um formato de saída específico (como JSON, XML ou código com anotações), inclua 1-2 exemplos de entrada/saída no prompt para guiar o modelo [1].
**3. Exija a Explicação (Chain-of-Thought):** Peça à IA para "Pensar passo a passo" ou "Explicar a lógica antes de fornecer o código". Isso melhora a precisão e permite a depuração do raciocínio [3].
**4. Validação e Testes:** Sempre solicite que o código gerado inclua testes unitários (JUnit, TestNG) e que a IA valide a solução contra casos de borda. O código gerado deve ser tratado como um rascunho que precisa de revisão humana [4].
**5. Controle de Versão e Dependências:** Especifique as versões exatas do Java, Spring, Maven/Gradle e outras bibliotecas para evitar alucinações de dependências ou incompatibilidades de sintaxe [5].

## Use Cases
**1. Geração Rápida de Boilerplate:** Criar classes de modelo (POJOs), controladores REST, serviços e repositórios em frameworks como Spring Boot ou Jakarta EE.
**2. Migração e Atualização de Código:** Auxiliar na migração de código Java antigo para novas versões (ex: Java 8 para Java 21) ou na refatoração para usar novos recursos de linguagem (ex: Records, Pattern Matching, API Stream) [1].
**3. Análise e Depuração:** Fornecer um *stack trace* e o código relacionado para que a IA identifique a causa raiz de um erro e sugira uma correção.
**4. Desenvolvimento Orientado a Testes (TDD):** Gerar testes unitários (JUnit, Mockito) para uma classe existente ou sugerir a implementação de um método com base em um teste fornecido.
**5. Documentação Automática:** Gerar documentação Javadoc detalhada para classes e métodos, especialmente em projetos legados com documentação esparsa.
**6. Geração de Código Específico para IA:** Usar LLMs para gerar código que interage com outros modelos de IA, como a integração de *ChatClients* ou a manipulação de dados para *Retrieval-Augmented Generation (RAG)*, como visto no Spring AI [1].

## Pitfalls
**1. Alucinação de APIs e Dependências:** A IA pode inventar métodos, classes ou dependências que não existem ou estão obsoletas. **Contramedida:** Sempre especifique a versão exata do framework (ex: Spring Boot 3.2, Java 21) e verifique o código gerado em um IDE [5].
**2. Geração de Código Inseguro:** LLMs podem gerar código com vulnerabilidades de segurança (ex: injeção de SQL, XSS) se não forem explicitamente instruídos a seguir práticas de codificação segura. **Contramedida:** Inclua sempre a instrução "Garanta que o código seja seguro e siga as diretrizes OWASP" [4].
**3. Falta de Contexto de Projeto:** O código gerado pode não se integrar corretamente com a arquitetura existente (ex: padrões de nomenclatura, injeção de dependência). **Contramedida:** Forneça trechos de código vizinhos ou a estrutura da classe para contextualizar o pedido [2].
**4. Prompts Vagos ou Ambíguos:** Pedidos como "Escreva um código Java para conectar ao banco de dados" são muito amplos. **Contramedida:** Seja ultra-específico: "Escreva um método Java usando JPA/Hibernate para buscar um 'Cliente' por 'id' em um repositório Spring Data JPA" [3].
**5. Confiança Excessiva (Over-reliance):** Tratar o código gerado pela IA como final sem revisão. **Contramedida:** O código da IA é um rascunho. Sempre revise, teste e depure manualmente [4].

## URL
[https://spring.io/blog/2025/04/14/spring-ai-prompt-engineering-patterns](https://spring.io/blog/2025/04/14/spring-ai-prompt-engineering-patterns)
