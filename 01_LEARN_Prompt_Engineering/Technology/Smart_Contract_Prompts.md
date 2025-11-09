# Smart Contract Prompts

## Description
**Smart Contract Prompts** são técnicas de Engenharia de Prompt focadas na utilização de Large Language Models (LLMs) para auxiliar no ciclo de vida de desenvolvimento de contratos inteligentes, desde a geração de código até a análise de segurança e detecção de vulnerabilidades.

Esta técnica é crucial devido à natureza imutável dos contratos inteligentes em blockchains, onde erros de codificação podem resultar em perdas financeiras catastróficas. Os prompts são estruturados para guiar o LLM a gerar código Solidity seguro, otimizado e aderente a padrões, ou para atuar como um auditor de segurança, aplicando raciocínio estruturado (como o *agent-role chaining* e *Chain-of-Thought*) para identificar falhas.

A eficácia dos Smart Contract Prompts reside na capacidade de fornecer contexto técnico profundo, requisitos de segurança explícitos e padrões de teste, transformando o LLM em uma ferramenta poderosa para a automação e garantia de qualidade no desenvolvimento Web3. Modelos de linguagem especializados, como o **Solidity-LLM**, são ajustados especificamente para essa finalidade.

## Examples
```
**1. Geração de Contrato ERC-20 Seguro (Geração de Código)**

```
Crie um contrato inteligente ERC-20 em Solidity (versão 0.8.20+) chamado "ManuToken" com um suprimento total de 1.000.000 de tokens. O contrato deve:
1. Implementar o padrão OpenZeppelin ERC20 e Ownable.
2. Incluir uma função `mint(address to, uint256 amount)` restrita apenas ao proprietário.
3. Implementar o `ReentrancyGuard` em todas as funções que manipulam Ether ou tokens externos (se aplicável).
4. Gerar um teste unitário básico usando o framework Foundry (Forge) para verificar a cunhagem inicial e a transferência de tokens.
5. Seguir o padrão Checks-Effects-Interactions (CEI).
```

**2. Análise de Vulnerabilidade (Detecção de Falhas)**

```
Você é um auditor de segurança sênior. Analise o seguinte código Solidity para vulnerabilidades de Reentrancy, Integer Overflow/Underflow e Time-Manipulation.
Use o raciocínio Chain-of-Thought (CoT) para explicar passo a passo como você chegaria à conclusão.
[INSERIR CÓDIGO VULNERÁVEL AQUI]
Se uma vulnerabilidade for encontrada, forneça o código corrigido e um exemplo de exploit.
```

**3. Geração de Teste de Fuzzing (Garantia de Qualidade)**

```
Para o contrato inteligente [NOME DO CONTRATO] com a função `deposit(uint256 amount)`, gere um teste de fuzzing usando o framework Foundry (Forge).
O teste deve cobrir os seguintes cenários:
1. Valores de `amount` positivos e negativos (se o tipo permitir).
2. Tentativas de depósito com `amount` zero.
3. Tentativas de depósito por um endereço sem permissão (se houver controle de acesso).
4. Verifique se o saldo do usuário e o suprimento total são atualizados corretamente após o depósito.
```

**4. Refatoração e Otimização de Gás (Otimização)**

```
Refatore o seguinte trecho de código Solidity para otimizar o consumo de gás, mantendo a funcionalidade original. Explique as otimizações realizadas.
[INSERIR TRECHO DE CÓDIGO AQUI]
Considere o uso de `calldata` em vez de `memory` para parâmetros de função externa e o armazenamento eficiente de variáveis de estado.
```

**5. Prompt Estruturado para Desenvolvimento Completo (Foundry)**

```
<system_context>
Você é um assistente avançado especializado em desenvolvimento de contratos inteligentes Ethereum com o framework Foundry.
</system_context>

<behavior_guidelines>
- Responda de forma clara e profissional.
- Foque exclusivamente em soluções e ferramentas baseadas em Foundry.
- Forneça exemplos de código completos e funcionais.
- Priorize segurança e eficiência de gás.
</behavior_guidelines>

<user_prompt>
Crie um contrato de cofre (Vault) compatível com ERC-4626. O cofre deve ter uma taxa de depósito de 0.5% e um mecanismo de distribuição de recompensas simples (apenas o proprietário pode chamar `distributeRewards()`). Inclua testes unitários e um teste de invariante para garantir que `totalSupply == sum of balances`.
</user_prompt>
```
```

## Best Practices
**1. Contextualização Detalhada:** Forneça o máximo de detalhes possível sobre a funcionalidade, o padrão (ex: ERC-20, ERC-721, ERC-4626) e o ambiente de desenvolvimento (ex: Foundry, Hardhat).
**2. Especificação de Segurança:** Inclua explicitamente requisitos de segurança, como "implementar ReentrancyGuard", "validar todas as entradas do usuário" e "seguir o padrão Checks-Effects-Interactions (CEI)".
**3. Requisitos de Teste:** Exija a inclusão de testes abrangentes (unitários, fuzzing, invariantes) e especifique o framework de teste (ex: Forge).
**4. Definição de Papéis (Role-Chaining):** Para análise de vulnerabilidades, defina papéis claros para o LLM (ex: "Você é um auditor de segurança sênior") e use prompts de raciocínio estruturado (Chain-of-Thought).
**5. Convenções de Nomenclatura:** Peça para seguir as convenções de nomenclatura do Solidity (PascalCase para contratos, mixedCase para funções e variáveis).

## Use Cases
**1. Geração Rápida de Protótipos:** Criar rapidamente contratos inteligentes baseados em padrões (ex: ERC-20, NFT) para testes e prova de conceito.
**2. Detecção e Correção de Vulnerabilidades:** Utilizar o LLM como um assistente de auditoria para analisar código existente, identificar falhas de segurança (ex: reentrancy, race conditions) e sugerir correções.
**3. Geração de Testes Automatizados:** Gerar suítes de testes abrangentes (unitários, de integração, fuzzing e invariantes) para garantir a robustez do código.
**4. Otimização de Gás:** Refatorar funções para reduzir o custo de transação na blockchain, um fator crítico para a usabilidade e economia.
**5. Documentação e Explicação de Código:** Gerar documentação NatSpec para funções e variáveis, ou pedir ao LLM para explicar a lógica de contratos complexos.
**6. Tradução de Requisitos:** Converter especificações de negócios em linguagem natural diretamente em código Solidity funcional.

## Pitfalls
**1. Confiança Cega (Blind Trust):** O maior erro é implantar o código gerado pelo LLM sem uma auditoria de segurança manual ou automatizada completa. LLMs podem gerar código funcional, mas com vulnerabilidades sutis.
**2. Falta de Contexto:** Prompts vagos ou incompletos levam a contratos que não atendem aos requisitos de negócios ou que falham em considerar casos de borda críticos.
**3. Ignorar Padrões de Teste:** Não exigir testes (unitários, fuzzing, invariantes) no prompt resulta em código não verificado e propenso a erros.
**4. Não Especificar a Versão do Solidity:** A omissão da versão do compilador pode levar a problemas de compatibilidade ou a vulnerabilidades corrigidas em versões mais recentes.
**5. Não Especificar o Framework:** Sem especificar o framework (Foundry, Hardhat), o LLM pode gerar código ou testes incompatíveis com o ambiente de desenvolvimento do usuário.
**6. Ignorar MEV/Front-Running:** Não incluir requisitos para mitigar ataques de MEV (Maximal Extractable Value) e front-running em prompts para contratos DeFi.

## URL
[https://getfoundry.sh/introduction/prompting/](https://getfoundry.sh/introduction/prompting/)
