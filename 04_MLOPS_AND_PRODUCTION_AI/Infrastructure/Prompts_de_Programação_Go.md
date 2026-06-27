# Prompts de Programação Go

## Description
**Prompts de Programação Go** (ou Golang) referem-se a instruções de engenharia de prompt otimizadas para interagir com Modelos de Linguagem Grande (LLMs) com o objetivo de gerar, analisar, depurar ou refatorar código na linguagem Go. Esta técnica se concentra em aproveitar a IA para tarefas específicas de desenvolvimento em Go, como a criação de microsserviços concorrentes, a aplicação de padrões de design idiomáticos e a resolução de problemas complexos de concorrência, como *deadlocks* e *goroutine leaks*.

A eficácia desses prompts reside na inclusão de contexto específico de Go, como a necessidade de código **idiomático** (seguindo o *Effective Go*), a manipulação explícita de erros e o uso correto de recursos de concorrência (goroutines e canais). Ao fornecer diretrizes claras sobre a arquitetura, as bibliotecas (ex: Gin, gRPC) e as convenções de Go, o desenvolvedor maximiza a precisão e a utilidade do código gerado pela IA.

## Examples
```
1.  **Geração de Microsserviço RESTful:**
    > "Você é um engenheiro de software sênior com vasta experiência em Go. Desenvolva um tutorial passo a passo, incluindo exemplos de código completos e explicativos, que demonstre como construir um microsserviço RESTful em Go para gerenciar usuários, utilizando o framework **Gin** e uma conexão com banco de dados **PostgreSQL**. O código deve seguir o estilo Go idiomático, com tratamento de erros explícito e a estrutura de projeto recomendada."

2.  **Debugging de Concorrência:**
    > "Você é um especialista em Go. Analise o seguinte trecho de código Go (inserir código) que está apresentando um *deadlock* intermitente. Crie um guia de depuração interativo para desenvolvedores de nível intermediário, explicando a causa raiz do problema e fornecendo a solução corrigida. O guia deve incluir as etapas para usar o **race detector** e o **pprof** para diagnosticar o problema."

3.  **Aplicação de Padrão de Design:**
    > "Crie um guia detalhado para desenvolvedores Go, explorando o padrão de design **Factory** para a criação de diferentes tipos de conexões de banco de dados (MySQL, MongoDB). Inclua exemplos de código em Go que ilustrem a aplicação do padrão de forma idiomática, focando na interface e na injeção de dependência."

4.  **Estratégia de Teste de Integração:**
    > "Elabore um tutorial passo a passo detalhado para iniciantes sobre como implementar testes de integração eficazes para um serviço Go que interage com um serviço externo de pagamento. O tutorial deve usar **test containers** para simular o serviço externo e garantir alta cobertura e confiabilidade, com exemplos de código para a configuração e execução dos testes."

5.  **Otimização de Performance:**
    > "Você é um especialista em otimização de performance em Go. Analise o seguinte código (inserir código) e forneça um tutorial prático sobre como otimizar o uso de memória e evitar alocações desnecessárias, focando no uso eficiente de *slices* e *maps*. Forneça a versão otimizada do código e explique as melhorias."

6.  **Geração de Documentação de API:**
    > "Crie um guia abrangente e prático para o desenvolvimento de documentação de API em projetos Go. O guia deve focar na geração automática de documentação a partir de comentários de código, utilizando a ferramenta **Swag**. Inclua exemplos de comentários de código formatados corretamente e os comandos necessários para gerar e hospedar a documentação."

7.  **Simulação de Sistema Complexo:**
    > "Crie um módulo Go que simule a gestão de um e-commerce. O módulo deve incluir structs para `Produto` (ID, Nome, Preço, QuantidadeEmEstoque) e `Pedido` (ID, Slice de IDs de Produtos, Status, DataCriacao). O prompt deve solicitar a implementação de uma função `ProcessarPedido` que utilize **goroutines** e **canais** para simular o processamento assíncrono de pedidos e a atualização do estoque."
```

## Best Practices
*   **Seja Idiomático:** Sempre instrua a IA a seguir o estilo Go idiomático, referenciando o *Effective Go* e as convenções de nomenclatura (`gofmt`).
*   **Especifique a Concorrência:** Ao lidar com tarefas concorrentes, seja explícito sobre o uso de **goroutines** e **canais**, e solicite a inclusão de mecanismos de sincronização (ex: `sync.Mutex`, `sync.WaitGroup`).
*   **Tratamento de Erros Explícito:** Exija que o código gerado utilize o tratamento de erros explícito de Go, retornando erros em vez de usar exceções ou *panics* (exceto em casos de erro irrecuperável).
*   **Defina o Contexto e as Dependências:** Especifique as bibliotecas, frameworks (ex: Gin, gRPC) e a versão do Go a serem utilizadas.
*   **Solicite Testes:** Peça à IA para gerar testes unitários e de integração para o código, seguindo o pacote `testing` padrão de Go.

## Use Cases
*   **Desenvolvimento Rápido de Protótipos:** Geração de *scaffolding* para microsserviços, APIs e ferramentas de linha de comando (CLI).
*   **Resolução de Problemas de Concorrência:** Diagnóstico e correção de *deadlocks*, *race conditions* e *goroutine leaks*.
*   **Refatoração e Otimização:** Sugestões para refatorar código não idiomático ou otimizar o uso de memória e CPU.
*   **Aprendizado e Tutoriais:** Criação de exemplos de código para padrões de design, estruturas de dados e recursos avançados de Go.
*   **Geração de Documentação:** Criação de documentação de API e guias de uso a partir do código-fonte.

## Pitfalls
*   **Código Não Idiomático:** A IA pode gerar código que funciona, mas não segue as convenções de Go (ex: uso excessivo de classes/herança em vez de composição, manipulação de erros não idiomática).
*   **Concorrência Incorreta:** A concorrência é complexa; a IA pode introduzir *race conditions* ou *deadlocks* sutis se o prompt não for rigoroso o suficiente sobre a sincronização.
*   **Dependência Excessiva:** Confiar cegamente no código gerado sem uma revisão crítica, especialmente em relação à segurança e performance.
*   **Falta de Contexto:** A IA pode não ter o contexto completo do projeto, resultando em código que não se integra bem com a arquitetura existente.
*   **Alucinações de Biblioteca:** A IA pode referenciar bibliotecas ou funções que estão obsoletas ou não existem mais na versão atual do Go.

## URL
[https://www.cabare.dev.br/topics/go](https://www.cabare.dev.br/topics/go)
