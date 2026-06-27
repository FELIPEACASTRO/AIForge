# JavaScript Programming Prompts

## Description
**Prompts de Programação JavaScript** referem-se a técnicas de engenharia de prompt otimizadas para interagir com Modelos de Linguagem Grande (LLMs) e ferramentas de assistência de codificação (como GitHub Copilot, Amazon CodeWhisperer ou Gemini Code Assist) com o objetivo de gerar, depurar, refatorar, documentar ou testar código JavaScript (incluindo Node.js e frameworks como React, Vue e Angular). A eficácia desses prompts reside na capacidade de fornecer contexto técnico preciso, definir o papel da IA, especificar a versão da linguagem (ex: ES6+), o ambiente de execução (ex: browser ou Node.js) e as bibliotecas/frameworks a serem utilizados. O foco é transformar a IA em um **engenheiro de software sênior** que segue as melhores práticas da comunidade JavaScript, como o uso de funções assíncronas (`async/await`), módulos ES, JSDoc para tipagem e testes unitários com Jest ou Mocha. A técnica é fundamental para aumentar a produtividade do desenvolvedor, automatizando tarefas repetitivas e acelerando a prototipagem, ao mesmo tempo que mantém a qualidade e a segurança do código gerado.

## Examples
```
**1. Geração de Função Assíncrona com JSDoc:**
```
"Você é um engenheiro de software Node.js sênior. Crie uma função assíncrona chamada `fetchUserData` que recebe um `userId` (string) e faz uma requisição GET para a API `https://api.example.com/users/{userId}`. A função deve usar `fetch` e retornar o objeto JSON do usuário. Inclua JSDoc completo para tipagem e descrição."
```

**2. Refatoração para ES6+:**
```
"Refatore o seguinte trecho de código JavaScript (ES5) para usar sintaxe moderna ES6+, incluindo `const`/`let`, arrow functions e template literals. O código deve ser mais conciso e legível.

[INSERIR CÓDIGO ES5 AQUI]"
```

**3. Geração de Teste Unitário (Jest):**
```
"Atue como um especialista em testes Jest. Crie um arquivo de testes unitários para a função JavaScript abaixo. Inclua casos de teste para sucesso, falha de validação e tratamento de exceções.

[INSERIR FUNÇÃO JAVASCRIPT AQUI]"
```

**4. Depuração e Correção de Erro:**
```
"Analise o seguinte código JavaScript e a mensagem de erro. Identifique a causa do erro (TypeError: Cannot read properties of undefined) e forneça o código corrigido, explicando a mudança.

Código: [INSERIR CÓDIGO COM ERRO AQUI]
Erro: [INSERIR MENSAGEM DE ERRO AQUI]"
```

**5. Criação de Componente React Funcional:**
```
"Crie um componente React funcional (usando TypeScript) chamado `UserProfileCard`. Ele deve aceitar um objeto `user` como prop (com campos `name: string`, `email: string`, `isActive: boolean`). O componente deve exibir o nome em um `<h1>` e o status de ativo com um badge verde/vermelho. Use hooks e siga as melhores práticas de React."
```

**6. Explicação de Conceito Complexo:**
```
"Explique o mecanismo de 'Event Loop' no Node.js para um desenvolvedor júnior que entende de JavaScript síncrono. Use analogias e forneça um pequeno exemplo de código para ilustrar a diferença entre a fila de microtasks e a fila de macrotasks."
```

**7. Geração de Script de Automação (Node.js):**
```
"Crie um script Node.js que leia um arquivo CSV (`data.csv`), itere sobre cada linha e faça uma requisição POST assíncrona para o endpoint `https://api.example.com/process-data` com os dados da linha. O script deve usar a biblioteca `axios` e limitar a 5 requisições paralelas para evitar sobrecarga."
```
```

## Best Practices
**1. Defina o Papel e o Contexto (Role and Context):** Comece o prompt definindo o papel da IA (ex: "Você é um engenheiro de software JavaScript sênior") e o contexto do projeto (ex: "Em um projeto Node.js com Express e TypeScript..."). Isso direciona o estilo e a complexidade da resposta.
**2. Seja Específico e Modular (Specific and Modular):** Solicite tarefas pequenas e bem definidas. Em vez de "Crie um backend", peça "Crie uma função `validateUser` que use `joi` para validar um objeto de usuário com `name` (string, obrigatório) e `age` (número, opcional)".
**3. Forneça Restrições e Padrões (Constraints and Standards):** Inclua requisitos de estilo, performance e segurança. Ex: "Use apenas sintaxe ES6+ e `async/await`", "O código deve ser otimizado para performance em loops grandes", "Garanta que não haja vulnerabilidades de injeção de dependência".
**4. Use a Abordagem de Diálogo (Dialogue Approach):** Use prompts de acompanhamento (follow-up) para refinar o código, solicitar testes unitários, documentação (JSDoc) ou refatoração. Isso simula um ciclo de desenvolvimento interativo.
**5. Inclua Exemplos de Código (Code Examples):** Para tarefas complexas ou específicas, inclua um pequeno trecho de código ou a assinatura da função esperada para guiar a IA.
**6. Valide e Revise (Validate and Review):** Sempre trate o código gerado pela IA como um rascunho. Revise-o, teste-o e integre-o ao seu sistema de controle de versão. A IA é um copiloto, não um piloto automático.

## Use Cases
**1. Desenvolvimento Acelerado de Funcionalidades:** Gerar rapidamente funções utilitárias, scripts de automação (Node.js) ou componentes de UI (React/Vue) a partir de especificações de alto nível.
**2. Refatoração e Modernização de Código:** Converter código legado (ES5) para padrões modernos (ES6+, TypeScript) e aplicar padrões de design (ex: Factory, Observer).
**3. Geração de Testes e Documentação:** Criar automaticamente testes unitários (Jest, Mocha) e documentação técnica (JSDoc, TypeDoc) para funções existentes, garantindo a qualidade do código.
**4. Depuração e Otimização de Performance:** Identificar e corrigir *bugs* em trechos de código complexos ou sugerir otimizações de performance para gargalos (ex: loops, operações assíncronas).
**5. Aprendizado e Explicação de Conceitos:** Usar a IA como um tutor para explicar conceitos complexos de JavaScript (ex: *closures*, *prototypes*, *Event Loop*) com exemplos de código prático e didático.

## Pitfalls
**1. Falta de Especificidade no Contexto:** Gerar código sem especificar o ambiente (Node.js, Browser, Deno) ou a versão da linguagem (ES5 vs. ES6+) leva a código incompatível ou desatualizado.
**2. Confiança Cega na Segurança:** O código gerado pela IA pode conter vulnerabilidades de segurança (ex: XSS, injeção de dependência) ou usar bibliotecas descontinuadas. **Sempre** realize uma revisão de segurança.
**3. Geração de Código "Boilerplate" Excessivo:** Prompts vagos resultam em código genérico e verboso, aumentando o *technical debt* (dívida técnica). A IA tende a ser prolixa se não for instruída a ser concisa.
**4. Ignorar a Arquitetura Existente:** A IA não tem conhecimento do seu código-base completo. Solicitar uma nova funcionalidade sem fornecer o contexto arquitetural (ex: como a injeção de dependência é feita) pode gerar código que não se integra.
**5. Prompts de Tarefas Múltiplas e Complexas:** Pedir para a IA "Criar um sistema de login completo com React, Express e MongoDB" em um único prompt resultará em uma resposta superficial e incompleta. Use a abordagem de **Chain-of-Thought** (Cadeia de Pensamento) e divida a tarefa em prompts sequenciais.

## URL
[https://treinamentosaf.com.br/prompts-para-geracao-de-codigo-python-e-javascript-guia-pratico-2025/](https://treinamentosaf.com.br/prompts-para-geracao-de-codigo-python-e-javascript-guia-pratico-2025/)
