# Peer Review Prompts

## Description
A técnica **Peer Review Prompts** (Prompts de Revisão por Pares) consiste em instruir um Large Language Model (LLM) a assumir o papel de um revisor ou crítico especializado para analisar um artefato (texto, código, design, etc.) e fornecer feedback estruturado, construtivo e acionável. Em vez de apenas pedir uma "revisão", o prompt define explicitamente a **persona do revisor**, os **critérios de avaliação** (checklist), o **formato de saída** desejado (e.g., pontos fortes, pontos fracos, sugestões de melhoria) e o **foco da análise** (e.g., coesão, segurança, usabilidade). Isso transforma o LLM de um gerador de conteúdo para um assistente de avaliação crítica, permitindo que o usuário desenvolva seu próprio julgamento retórico ao comparar o feedback da IA com o de revisores humanos. É uma técnica fundamental para aprimorar a qualidade de textos, códigos e designs, utilizando a IA como um par de olhos experiente e metódico.

## Examples
```
**Exemplo 1: Revisão de Ensaio Acadêmico**
```
**Persona:** Você é um Professor de Retórica e Composição com 20 anos de experiência.
**Tarefa:** Revise o ensaio do aluno abaixo e forneça feedback construtivo.
**Checklist:**
1.  **Tese:** A tese é clara, específica e defensável?
2.  **Evidência:** Cada ponto principal é suportado por evidências sólidas e citadas corretamente?
3.  **Coerência:** A transição entre os parágrafos é suave e a estrutura lógica é clara?
**Formato de Saída:**
1.  **Resumo da Avaliação:** (Nota geral e ponto principal de melhoria)
2.  **Comentários Específicos:** (3-5 observações numeradas, citando a linha ou seção do texto)
3.  **Sugestão de Revisão:** (Uma ação acionável para a próxima etapa)

[COLE O ENSAIO AQUI]
```

**Exemplo 2: Revisão de Código (Python)**
```
**Persona:** Você é um Engenheiro de Software Sênior especializado em Python e segurança.
**Tarefa:** Analise o trecho de código Python para segurança, performance e aderência ao PEP 8.
**Checklist:**
1.  **Segurança:** Existem vulnerabilidades como injeção de SQL, exposição de segredos ou uso inseguro de bibliotecas?
2.  **Performance:** Há loops ineficientes, consultas desnecessárias ou oportunidades de otimização de complexidade O(n)?
3.  **Estilo (PEP 8):** O código segue as convenções de nomenclatura e formatação do PEP 8?
**Formato de Saída:**
1.  **Pontos Críticos (Bloqueadores):** (Erros de segurança ou bugs)
2.  **Sugestões de Refatoração:** (Melhorias de performance e clareza)
3.  **Comentários de Estilo:** (Ajustes de formatação com base no PEP 8)

[COLE O CÓDIGO PYTHON AQUI]
```

**Exemplo 3: Crítica de Design UX/UI**
```
**Persona:** Você é um Especialista em UX/UI focado em acessibilidade e usabilidade móvel.
**Tarefa:** Revise o wireframe (descrito abaixo) para a tela de checkout de um aplicativo de e-commerce.
**Checklist:**
1.  **Usabilidade:** O fluxo de checkout é intuitivo e minimiza a carga cognitiva?
2.  **Acessibilidade (WCAG):** O contraste de cores é adequado e os elementos de toque são grandes o suficiente?
3.  **Consistência:** O design segue padrões de interface conhecidos e consistentes com o restante do aplicativo?
**Formato de Saída:**
1.  **Problemas de Usabilidade (Prioridade Alta):** (Descreva o problema e o impacto no usuário)
2.  **Melhorias de Acessibilidade:** (Sugestões específicas com base em WCAG)
3.  **Elogios:** (O que funciona bem no design)

[DESCRIÇÃO DO WIREFRAME AQUI]
```

**Exemplo 4: Revisão de Documentação Técnica**
```
**Persona:** Você é um Editor Técnico com foco em clareza e precisão para um público de desenvolvedores juniores.
**Tarefa:** Revise o seguinte trecho de documentação da API.
**Checklist:**
1.  **Precisão Técnica:** Os exemplos de código e as descrições dos parâmetros estão 100% corretos e atualizados?
2.  **Clareza e Simplicidade:** A linguagem é direta e evita jargões desnecessários? O público júnior entenderá?
3.  **Estrutura:** O uso de títulos, listas e blocos de código está formatado corretamente para facilitar a leitura?
**Formato de Saída:**
1.  **Pontos Fortes:** (Onde a documentação é excelente)
2.  **Revisões Necessárias:** (Sugestões de reescrita para clareza, com a frase original e a sugestão)
3.  **Verificação de Fatos:** (Quaisquer afirmações técnicas que precisam de dupla checagem)

[COLE O TRECHO DA DOCUMENTAÇÃO AQUI]
```

**Exemplo 5: Revisão de Prompt (Metaprompting)**
```
**Persona:** Você é um Engenheiro de Prompt Sênior.
**Tarefa:** Analise o prompt abaixo para clareza, especificidade e potencial risco de "prompt injection" ou ambiguidade.
**Checklist:**
1.  **Clareza da Intenção:** O objetivo do prompt é inequívoco?
2.  **Especificidade do Formato:** O formato de saída é claramente definido e restritivo?
3.  **Risco de Injeção:** Há alguma parte do prompt que possa ser explorada por um usuário mal-intencionado para desviar a IA?
**Formato de Saída:**
1.  **Diagnóstico:** (Avaliação geral: Bom, Precisa de Ajustes, Ruim)
2.  **Melhorias Sugeridas:** (Como tornar o prompt mais robusto e específico)
3.  **Alerta de Risco:** (Se houver risco de injeção, descreva como mitigá-lo)

[COLE O PROMPT A SER REVISADO AQUI]
```
```

## Best Practices
1. **Definir a Persona e o Papel:** Comece o prompt instruindo o LLM a assumir um papel específico (e.g., "Você é um Engenheiro de Software Sênior", "Você é um Revisor Cego de uma Revista Acadêmica de Alto Impacto", "Você é um Designer UX/UI focado em acessibilidade"). 2. **Fornecer Critérios Claros (Checklist):** Inclua uma lista numerada ou com marcadores dos pontos exatos que o LLM deve verificar. Isso garante que a revisão seja focada e abrangente. 3. **Estrutura de Saída (Output Structure):** Especifique o formato exato da resposta (e.g., "Use os seguintes cabeçalhos: [1. Resumo Geral], [2. Pontos Fortes], [3. Pontos Fracos/Áreas de Melhoria], [4. Sugestões Acionáveis]"). 4. **Análise em Partes:** Para documentos longos (artigos, grandes blocos de código), peça ao LLM para revisar seção por seção ou parágrafo por parágrafo para manter a precisão e evitar a perda de contexto. 5. **Incentivar a Justificativa:** Peça ao LLM para justificar cada crítica ou sugestão, citando a parte do texto ou código que levou à observação.

## Use Cases
1. **Revisão Acadêmica e de Escrita:** Avaliação de ensaios, teses e artigos de pesquisa quanto à clareza da tese, estrutura lógica, uso de evidências, tom e estilo de citação. 2. **Revisão de Código (Code Review):** Análise de trechos de código para identificar bugs, vulnerabilidades de segurança, aderência a padrões de codificação (PEP 8, etc.), complexidade e oportunidades de refatoração. 3. **Crítica de Design (Design Critique):** Avaliação de wireframes, mockups ou protótipos de design UX/UI quanto à usabilidade, acessibilidade, consistência visual e alinhamento com os objetivos do usuário. 4. **Revisão de Documentação Técnica:** Verificação de manuais, FAQs ou documentação de API quanto à precisão técnica, clareza e adequação ao público-alvo. 5. **Desenvolvimento de Prompts (Metaprompting):** Usar o LLM para revisar e refinar outros prompts, avaliando sua clareza, especificidade e potencial para injeção de prompt.

## Pitfalls
1. **Revisão Superficial:** Sem um checklist detalhado, o LLM pode fornecer feedback genérico e inútil. 2. **Foco Excessivo na Forma:** O LLM pode se concentrar demais em aspectos gramaticais ou estilísticos, ignorando falhas conceituais ou lógicas profundas. 3. **Vieses da IA:** O LLM pode introduzir vieses (e.g., favorecer o Inglês Padrão Branco, ou estilos de codificação populares) ou enfraquecer a análise crítica humana. 4. **Injeção de Prompt Oculta:** Em contextos de revisão de documentos de terceiros (como em revistas acadêmicas), pode haver texto adversário oculto no documento para manipular o feedback do LLM. 5. **Confiança Excessiva:** O usuário pode aceitar o feedback da IA sem aplicar seu próprio julgamento crítico, perdendo a oportunidade de aprendizado e revisão aprofundada.

## URL
[https://wac.colostate.edu/repository/collections/textgened/rhetorical-engagements/using-llms-as-peer-reviewers-for-revising-essays/](https://wac.colostate.edu/repository/collections/textgened/rhetorical-engagements/using-llms-as-peer-reviewers-for-revising-essays/)
