# Prompts de Programação Rust

## Description
Prompts de Programação Rust são instruções de engenharia de prompt altamente especializadas, projetadas para alavancar modelos de linguagem grandes (LLMs) na assistência ao desenvolvimento de software na linguagem Rust. Devido à natureza única do Rust, com seu sistema de *ownership* (posse), *borrowing* (empréstimo) e *lifetimes* (tempo de vida), prompts genéricos são ineficazes. Esta técnica foca em fornecer contexto detalhado e exigir que a IA atue como um especialista em Rust, abordando desafios específicos como otimização de performance (evitando alocações e cópias desnecessárias), refatoração idiomática (uso de `Result`/`Option`), e auditoria de segurança (uso seguro de `unsafe` e concorrência). O objetivo é transformar a IA de um assistente de codificação genérico em um co-piloto especializado que entende e respeita a filosofia de segurança e performance do Rust.

## Examples
```
**1. Otimização de Performance (Critical Path Analysis)**
**Role:** Atue como um Engenheiro de Performance Sênior em Rust.
**Task:** Analise o bloco de código Rust fornecido abaixo. Identifique o "caminho crítico" (hot path) e sugira otimizações que reduzam a alocação de memória, minimizem cópias desnecessárias e melhorem a eficiência do iterador. Otimize para latência, não apenas para throughput.
**Code:** [INSERIR CÓDIGO RUST AQUI]
**Constraint:** Mantenha a segurança de memória e a legibilidade do código.

**2. Refatoração e Idiomaticidade (Idiomatic Refactoring)**
**Role:** Atue como um Revisor de Código Rust experiente.
**Task:** Refatore o código Rust a seguir para torná-lo mais idiomático, utilizando padrões de design Rust (como `Result`, `Option`, `match`), e seguindo as diretrizes do `clippy`. Explique cada alteração e o princípio Rust que ela reforça.
**Code:** [INSERIR CÓDIGO RUST AQUI]
**Constraint:** O código refatorado deve passar no `cargo clippy -- -D warnings`.

**3. Auditoria de Segurança (Security Audit)**
**Role:** Atue como um Auditor de Segurança de Software especializado em Rust.
**Task:** Revise o módulo Rust fornecido em busca de vulnerabilidades de segurança, como erros de uso de `unsafe`, problemas de concorrência (data races), ou falhas na manipulação de entrada/saída (I/O). Sugira correções e explique a falha de segurança que cada correção mitiga.
**Code:** [INSERIR CÓDIGO RUST AQUI]
**Constraint:** Priorize a eliminação de qualquer uso de `unsafe` que não seja estritamente necessário.

**4. Geração de Testes de Unidade (Unit Test Generation)**
**Role:** Atue como um Engenheiro de Testes de Software.
**Task:** Gere um conjunto abrangente de testes de unidade para a função Rust fornecida. Inclua casos de teste para entradas válidas, casos de borda (edge cases), e manipulação de erros (`panic` ou `Result::Err`).
**Function:** [INSERIR FUNÇÃO RUST AQUI]
**Constraint:** Use a macro `#[cfg(test)]` e o módulo de teste padrão do Rust.

**5. Explicação de Erro de Compilação (Compiler Error Explanation)**
**Role:** Atue como um Tutor de Linguagem Rust.
**Task:** Explique o erro de compilação do Rust fornecido abaixo. Descreva a causa raiz do erro (por exemplo, regra de *ownership* violada, erro de *lifetime*), e forneça uma solução de código mínima e funcional.
**Error Message:** [INSERIR MENSAGEM DE ERRO DO COMPILADOR AQUI]
**Constraint:** A explicação deve ser clara e didática, focando no conceito central do Rust.

**6. Geração de Estrutura de Projeto (Project Structure Generation)**
**Role:** Atue como um Arquiteto de Software.
**Task:** Gere a estrutura de arquivos e diretórios (`Cargo.toml`, `src/main.rs`, `src/lib.rs`, módulos, etc.) para um novo projeto Rust que implementará um servidor web assíncrono usando `tokio` e `actix-web`.
**Project Goal:** Servidor RESTful para gerenciamento de usuários.
**Constraint:** O projeto deve seguir as convenções de módulos do Rust e estar pronto para ser compilado com `cargo build`.

**7. Documentação e Exemplo (Documentation and Example)**
**Role:** Atue como um Escritor Técnico.
**Task:** Escreva a documentação completa para a função Rust fornecida, incluindo um exemplo de uso no formato `doctest`. A documentação deve explicar claramente os parâmetros, o valor de retorno e o comportamento da função.
**Function:** [INSERIR FUNÇÃO RUST AQUI]
**Constraint:** Use a sintaxe de documentação padrão do Rust (`///`).
```

## Best Practices
**Seja Específico com a Função e o Contexto (Role and Context Specificity):** Sempre defina um papel especializado para a IA (ex: "Engenheiro de Performance Rust Sênior", "Auditor de Segurança"). Forneça o bloco de código completo ou a mensagem de erro do compilador para garantir que a IA tenha o contexto necessário para as regras de *ownership* e *lifetimes*. **Foque em Conceitos Idiomáticos:** Direcione a IA para usar padrões de design Rust, como `Result`, `Option`, `match`, e para seguir as diretrizes do `clippy`. Peça explicações sobre como as sugestões aderem à filosofia do Rust. **Validação Cruzada com Ferramentas Nativas:** Use os prompts para gerar código ou sugestões, mas sempre valide o resultado com as ferramentas nativas do Rust, como `cargo check`, `cargo clippy`, e testes de unidade (`cargo test`).

## Use Cases
**Otimização de Código de Alto Desempenho:** Identificar e corrigir gargalos de performance em código Rust, especialmente em loops, iterações e manipulação de coleções, focando em evitar alocações no *heap*. **Refatoração para Idiomaticidade:** Converter código funcional, mas não idiomático, em código que segue as melhores práticas e padrões do ecossistema Rust, melhorando a legibilidade e a manutenibilidade. **Auditoria de Segurança Automatizada:** Utilizar a IA para revisar módulos críticos, especialmente aqueles que usam `unsafe` ou manipulam concorrência, em busca de *data races* ou outras vulnerabilidades de segurança. **Geração de Testes Abrangentes:** Criar testes de unidade e integração robustos que cobrem casos de uso normais, casos de borda e manipulação de erros, garantindo a qualidade do software. **Aceleração da Curva de Aprendizado:** Usar a IA como um tutor especializado para explicar erros de compilação complexos relacionados a *ownership*, *borrowing* e *lifetimes*, acelerando o domínio da linguagem.

## Pitfalls
**Prompts Genéricos:** Usar prompts como "Otimize este código" sem especificar o que otimizar (memória, CPU, I/O) ou sem mencionar as regras de *ownership* do Rust. **Ignorar a Saída do Compilador:** Não fornecer a mensagem de erro completa do compilador do Rust (que é altamente informativa) à IA, resultando em correções incorretas ou incompletas. **Falta de Contexto de *Lifetime*:** Pedir à IA para refatorar código que envolve *lifetimes* complexos sem fornecer o contexto completo das funções e estruturas circundantes. **Confiar Cegamente em `unsafe`:** Permitir que a IA sugira o uso de blocos `unsafe` sem uma justificativa clara e uma explicação sobre como a segurança de memória é mantida, violando a filosofia central do Rust.

## URL
[https://github.com/Ranrar/rustic-prompt](https://github.com/Ranrar/rustic-prompt)
