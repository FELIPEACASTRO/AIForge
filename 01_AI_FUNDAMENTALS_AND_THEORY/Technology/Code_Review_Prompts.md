# Code Review Prompts

## Description
**Code Review Prompts** (Prompts de Revisão de Código) são instruções estruturadas e detalhadas fornecidas a um Large Language Model (LLM) para que ele execute uma análise crítica e construtiva de um trecho de código. O objetivo é automatizar ou auxiliar o processo de revisão de Pull Requests (PRs) ou commits, identificando problemas de qualidade, bugs, vulnerabilidades de segurança, ineficiências de desempenho e inconsistências de estilo [1] [2]. Essa técnica transforma o LLM de um simples gerador de texto em um assistente de engenharia de software, capaz de aplicar regras de negócio e padrões de codificação específicos, desde que o prompt forneça o contexto e as restrições necessárias [2]. A eficácia reside na capacidade de especificar o **papel** do LLM (ex: "Engenheiro Sênior de Segurança"), o **foco** da revisão (ex: "apenas performance") e o **formato** de saída desejado [1].

## Examples
```
**1. Revisão de Segurança (OWASP Top 10):**

```
Você é um Engenheiro Sênior de Segurança. Analise o código a seguir (linguagem: Python, framework: Django) estritamente sob a ótica das vulnerabilidades do OWASP Top 10. Para cada vulnerabilidade encontrada, forneça: a linha exata, uma explicação do risco e o código de correção sugerido. Se não houver vulnerabilidades, responda apenas 'Nenhuma vulnerabilidade de segurança crítica encontrada.'

[CÓDIGO AQUI]
```

**2. Revisão de Performance e Algoritmo:**

```
Atue como um especialista em otimização de performance. Revise a função [NOME DA FUNÇÃO] em [LINGUAGEM] para identificar gargalos de desempenho e ineficiências algorítmicas (complexidade O(n)). Sugira refatorações que melhorem a eficiência, justificando a nova complexidade. O foco é reduzir o uso de CPU e memória.

[CÓDIGO AQUI]
```

**3. Revisão de Estilo e Padrões de Código (Clean Code):**

```
Você é o revisor de código responsável por manter a consistência do nosso codebase. Avalie o código em relação aos princípios de Clean Code e ao padrão de nomenclatura CamelCase. Verifique: 1. Nomes de variáveis e funções são claros? 2. A função faz apenas uma coisa (Princípio da Responsabilidade Única)? 3. Há comentários desnecessários ou código morto? Retorne as sugestões em formato de lista numerada.

[CÓDIGO AQUI]
```

**4. Revisão de Testes Unitários (Test Coverage):**

```
O código a seguir é um novo módulo em [LINGUAGEM]. Revise os testes unitários fornecidos. Identifique quaisquer casos de borda (edge cases) que não foram cobertos e escreva os testes unitários adicionais necessários para atingir 100% de cobertura de linha. Use o framework [NOME DO FRAMEWORK DE TESTE].

[CÓDIGO AQUI]
```

**5. Revisão de Arquitetura e Manutenibilidade:**

```
Analise o código como um Arquiteto de Software. O objetivo desta revisão é garantir a manutenibilidade e a modularidade. O código adere ao padrão de projeto [NOME DO PADRÃO, ex: MVC]? Há alto acoplamento ou baixa coesão? Sugira refatorações para desacoplar componentes, se necessário.

[CÓDIGO AQUI]
```
```

## Best Practices
**Seja Específico e Contextual:** O prompt deve incluir o máximo de contexto possível, como o objetivo da mudança, o framework/linguagem, e as regras de estilo da equipe. Use a arquitetura **HITL (Human-in-the-Loop)**, onde a IA gera um rascunho de revisão e o revisor humano aprova, edita ou rejeita as sugestões, fornecendo um loop de feedback contínuo [2]. **Force o Formato de Saída:** Peça à LLM para retornar a revisão em um formato estruturado (como JSON ou Markdown com seções claras) para facilitar a análise e a integração em pipelines de CI/CD [2]. **Restrição de Escopo e Risco:** Para grandes bases de código, inclua apenas o diff e o contexto imediato (cabeçalho do arquivo, imports). Use tokens de risco (`HIGH_RISK`) para arquivos sensíveis (ex: criptografia, SQL) para que a LLM aplique heurísticas mais rigorosas [2]. **Verificação de Conformidade:** Incorpore listas de verificação obrigatórias (ex: "Nenhum segredo hard-coded", "Uso de `try-catch-finally`") diretamente no prompt para garantir a conformidade regulatória e de segurança [2].

## Use Cases
**Aceleração do Ciclo de Revisão:** Redução do tempo de espera por revisões, permitindo que a LLM trie e pré-aprove mudanças de baixo risco (ex: documentação) e sinalize apenas PRs de alto risco para revisão humana [2]. **Garantia de Consistência:** Aplicação uniforme de regras de estilo, padrões de nomenclatura e heurísticas de segurança em toda a base de código, superando a inconsistência entre revisores humanos [2]. **Mentoria Instantânea e Onboarding:** Fornecimento de feedback educacional instantâneo para desenvolvedores juniores, explicando o "porquê" por trás das sugestões de refatoração, acelerando a transferência de conhecimento [2]. **Detecção de Vulnerabilidades:** Combinação de análise estática com raciocínio de LLM para identificar vulnerabilidades de segurança (SQLi, XSS) e violações de conformidade (ex: GDPR) que linters tradicionais podem perder [2]. **Codificação de Conhecimento Tribal:** Incorporação de padrões internos e conhecimento tácito da equipe (ex: "sempre usar `await` em `fetch` neste repositório") no prompt da LLM, preservando a expertise mesmo com a saída de engenheiros sêniores [2].

## Pitfalls
**Falsos Positivos/Negativos:** A LLM pode sinalizar código inofensivo ou, pior, ignorar um bug crítico (alucinação), levando à **fadiga de alerta** e à desconfiança do revisor humano [2]. **Cegueira de Contexto:** O modelo pode sugerir mudanças que quebram contratos de domínio ou regras de negócio porque o prompt continha apenas o diff, e não o grafo de chamadas completo ou a configuração de tempo de execução [2]. **Vazamento de Segurança e Privacidade:** Enviar código sensível (ex: chaves de API) para LLMs hospedadas por terceiros pode violar a conformidade (GDPR, HIPAA) se não houver um Acordo de Processamento de Dados (DPA) adequado [2]. **Erosão de Habilidade:** A dependência excessiva da IA pode levar à **erosão das habilidades** de revisão e depuração dos desenvolvedores, especialmente os juniores, que param de aprender a ler erros de compilador e a pensar criticamente sobre o código [2]. **Viés e Estilo Obsoleto:** O modelo pode impor um estilo de codificação herdado de seu treinamento ou de um prompt desatualizado, causando atrito com as convenções em evolução da equipe [2].

## URL
[https://dev.to/dixitgurv/ai-assisted-code-review-opportunities-and-pitfalls-llp](https://dev.to/dixitgurv/ai-assisted-code-review-opportunities-and-pitfalls-llp)
