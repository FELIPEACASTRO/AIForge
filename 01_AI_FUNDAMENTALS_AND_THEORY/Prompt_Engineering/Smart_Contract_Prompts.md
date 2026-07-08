# Smart Contract Prompts

## Description
**Smart Contract Prompts** are Prompt Engineering techniques focused on using Large Language Models (LLMs) to assist in the smart contract development lifecycle, from code generation to security analysis and vulnerability detection.

This technique is crucial due to the immutable nature of smart contracts on blockchains, where coding errors can result in catastrophic financial losses. The prompts are structured to guide the LLM to generate secure, optimized, and standards-compliant Solidity code, or to act as a security auditor, applying structured reasoning (such as *agent-role chaining* and *Chain-of-Thought*) to identify flaws.

The effectiveness of Smart Contract Prompts lies in the ability to provide deep technical context, explicit security requirements, and testing standards, turning the LLM into a powerful tool for automation and quality assurance in Web3 development. Specialized language models, such as **Solidity-LLM**, are fine-tuned specifically for this purpose.

## Examples
```
**1. Secure ERC-20 Contract Generation (Code Generation)**

```
Create an ERC-20 smart contract in Solidity (version 0.8.20+) named "ManuToken" with a total supply of 1,000,000 tokens. The contract must:
1. Implement the OpenZeppelin ERC20 and Ownable standards.
2. Include a `mint(address to, uint256 amount)` function restricted to the owner only.
3. Implement `ReentrancyGuard` on all functions that handle Ether or external tokens (if applicable).
4. Generate a basic unit test using the Foundry (Forge) framework to verify the initial minting and token transfer.
5. Follow the Checks-Effects-Interactions (CEI) pattern.
```

**2. Vulnerability Analysis (Flaw Detection)**

```
You are a senior security auditor. Analyze the following Solidity code for Reentrancy, Integer Overflow/Underflow, and Time-Manipulation vulnerabilities.
Use Chain-of-Thought (CoT) reasoning to explain step by step how you would reach the conclusion.
[INSERT VULNERABLE CODE HERE]
If a vulnerability is found, provide the corrected code and an exploit example.
```

**3. Fuzzing Test Generation (Quality Assurance)**

```
For the smart contract [CONTRACT NAME] with the `deposit(uint256 amount)` function, generate a fuzzing test using the Foundry (Forge) framework.
The test must cover the following scenarios:
1. Positive and negative `amount` values (if the type allows).
2. Deposit attempts with a zero `amount`.
3. Deposit attempts from an address without permission (if access control exists).
4. Verify that the user balance and the total supply are updated correctly after the deposit.
```

**4. Gas Refactoring and Optimization (Optimization)**

```
Refactor the following Solidity code snippet to optimize gas consumption while maintaining the original functionality. Explain the optimizations performed.
[INSERT CODE SNIPPET HERE]
Consider using `calldata` instead of `memory` for external function parameters and the efficient storage of state variables.
```

**5. Structured Prompt for Complete Development (Foundry)**

```
<system_context>
You are an advanced assistant specialized in Ethereum smart contract development with the Foundry framework.
</system_context>

<behavior_guidelines>
- Respond clearly and professionally.
- Focus exclusively on Foundry-based solutions and tools.
- Provide complete and functional code examples.
- Prioritize security and gas efficiency.
</behavior_guidelines>

<user_prompt>
Create an ERC-4626-compatible Vault contract. The vault must have a 0.5% deposit fee and a simple rewards distribution mechanism (only the owner can call `distributeRewards()`). Include unit tests and an invariant test to ensure that `totalSupply == sum of balances`.
</user_prompt>
```
```

## Best Practices
**1. Detailed Contextualization:** Provide as much detail as possible about the functionality, the standard (e.g., ERC-20, ERC-721, ERC-4626), and the development environment (e.g., Foundry, Hardhat).
**2. Security Specification:** Explicitly include security requirements, such as "implement ReentrancyGuard", "validate all user inputs", and "follow the Checks-Effects-Interactions (CEI) pattern".
**3. Testing Requirements:** Require the inclusion of comprehensive tests (unit, fuzzing, invariant) and specify the testing framework (e.g., Forge).
**4. Role Definition (Role-Chaining):** For vulnerability analysis, define clear roles for the LLM (e.g., "You are a senior security auditor") and use structured reasoning prompts (Chain-of-Thought).
**5. Naming Conventions:** Ask it to follow Solidity naming conventions (PascalCase for contracts, mixedCase for functions and variables).

## Use Cases
**1. Rapid Prototype Generation:** Quickly create standards-based smart contracts (e.g., ERC-20, NFT) for testing and proof of concept.
**2. Vulnerability Detection and Correction:** Use the LLM as an auditing assistant to analyze existing code, identify security flaws (e.g., reentrancy, race conditions), and suggest fixes.
**3. Automated Test Generation:** Generate comprehensive test suites (unit, integration, fuzzing, and invariant) to ensure code robustness.
**4. Gas Optimization:** Refactor functions to reduce transaction cost on the blockchain, a critical factor for usability and economics.
**5. Documentation and Code Explanation:** Generate NatSpec documentation for functions and variables, or ask the LLM to explain the logic of complex contracts.
**6. Requirements Translation:** Convert business specifications in natural language directly into functional Solidity code.

## Pitfalls
**1. Blind Trust:** The biggest mistake is deploying LLM-generated code without a complete manual or automated security audit. LLMs can generate functional code, but with subtle vulnerabilities.
**2. Lack of Context:** Vague or incomplete prompts lead to contracts that do not meet business requirements or that fail to consider critical edge cases.
**3. Ignoring Testing Standards:** Not requiring tests (unit, fuzzing, invariant) in the prompt results in unverified and error-prone code.
**4. Not Specifying the Solidity Version:** Omitting the compiler version can lead to compatibility problems or to vulnerabilities that were fixed in newer versions.
**5. Not Specifying the Framework:** Without specifying the framework (Foundry, Hardhat), the LLM may generate code or tests incompatible with the user's development environment.
**6. Ignoring MEV/Front-Running:** Not including requirements to mitigate MEV (Maximal Extractable Value) and front-running attacks in prompts for DeFi contracts.

## URL
[https://getfoundry.sh/introduction/prompting/](https://getfoundry.sh/introduction/prompting/)
