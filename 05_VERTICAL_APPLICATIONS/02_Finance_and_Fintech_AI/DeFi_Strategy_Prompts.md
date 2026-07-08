# DeFi Strategy Prompts

## Description
**DeFi Strategy Prompts** are highly structured, contextual instructions provided to Large Language Models (LLMs) to simulate complex financial analyses, assess risks, and generate investment strategies in the Decentralized Finance (DeFi) ecosystem. They turn AI into a research and analysis assistant capable of processing information about protocols, *tokenomics*, *yield farming*, impermanent loss (IL), and smart-contract security. The effectiveness of these prompts lies in the ability to assign the AI a specialized role and request a multifaceted, objective analysis, helping the user make more informed and safer investment decisions in the volatile DeFi market [1].

## Examples
```
**1. Protocol and Risk Analysis**
`Act as a senior risk analyst. I am considering investing in the [Protocol Name] protocol on the [Network Name] network. Provide an objective analysis that includes: 1) Overview of how the mechanism works, 2) Tokenomics analysis (distribution, inflation/deflation), 3) Main technical risks (audits, admin keys) and economic risks (governance, *rug pull*), and 4) APY comparison with [Competing Protocol]. Conclude with a summary of Pros and Cons for an investment decision.`

**2. *Yield Farming* and Impermanent Loss Strategy**
`I am an experienced *yield farmer*. Explain the concept of Impermanent Loss (IL) in an accessible way. Then, analyze the liquidity pair [Token A]/[Token B] on the [DEX Name] DEX. What are the expected IL risks for a 25% price change in [Token A]? Suggest 3 strategies to mitigate IL for this pair, focusing on low-volatility pools or concentrated-liquidity solutions.`

**3. Smart-Contract Security Assessment**
`Act as a blockchain security auditor. Explain the 5 main attack vectors against DeFi smart contracts (e.g., *reentrancy*, *flash loans*). Then, for the [Protocol Name] protocol, research and summarize the status of its security audits (who audited, date, critical *findings* resolved). Provide practical tips for a lay investor to verify a contract's security before interacting with it.`

**4. Lending/Borrowing Optimization**
`I want to optimize my capital in [Token A] through lending/borrowing (*lending/borrowing*). Compare the platforms [Platform 1] and [Platform 2] based on: 1) APY rates for *lending* [Token A], 2) Interest rates for *borrowing* [Token B], 3) Collateralization factor and liquidation risk. Which platform offers the best risk-reward ratio for a conservative *looping* strategy?`

**5. Market Trends and Narratives Analysis**
`Act as a DeFi market-trends analyst. What are the 3 main investment narratives for the next quarter (e.g., *Restaking*, *Real World Assets - RWA*, specific *Layer 2*)? For the [Chosen Narrative] narrative, identify 2 promising projects and justify the choice based on technological innovation, community support, and TVL (Total Value Locked) growth potential.`
```

## Best Practices
**1. Be Specific and Contextual:** Clearly define the protocol, the token, the strategy (e.g., *yield farming*, *staking*, *lending*), and the objective (e.g., *maximize APY*, *minimize IL*, *risk analysis*). **2. Define the AI's Role:** Begin the prompt by assigning the AI a specialized role, such as "Act as a senior DeFi risk analyst" or "You are an experienced *yield farming* strategist". **3. Require Risk Analysis:** Always include a section that requests analysis of technical (smart contract), economic (tokenomics), and market (liquidity, volatility) risks. **4. Request a Structured Format:** Ask for the output in an easy-to-consume format, such as a comparison table, a pros-and-cons list, or an executive summary. **5. Use Current Data (If Possible):** If the AI platform allows, provide real-time data or ask the AI to fetch up-to-date information on APY, liquidity, and audits.

## Use Cases
**1. Protocol Risk Analysis:** Assess the security and economic sustainability of new DeFi protocols before investing. **2. Yield Strategy Optimization (*Yield Farming*):** Determine the best liquidity pairs, *lending* or *staking* platforms to maximize APY and manage Impermanent Loss (IL). **3. Education and Simulation:** Explain complex DeFi concepts (e.g., *tokenomics*, *liquidation*, *rebase*) and simulate market scenarios for educational purposes. **4. Smart-Contract Due Diligence:** Obtain a summary of security audits and the technical risks associated with a specific smart contract. **5. Opportunity Comparison:** Objectively compare different platforms (DEXs, *lending protocols*, *perpetuals*) based on specific metrics (fees, liquidity, security).

## Pitfalls
**1. Relying on Outdated Data:** LLMs have no native access to real-time data on APY, TVL, or token prices. The prompt should be built to request analysis of *structure* and *risk*, not price predictions or volatile data. **2. Generic Prompts:** Questions like "What's the best DeFi investment?" produce vague, useless answers. Lack of context (network, token, objective) is the most common mistake. **3. Ignoring the Source:** The AI may hallucinate or provide incorrect information about audits or *tokenomics*. The user should always use the prompt's output as a starting point for **manual verification** on the protocol's official website or with on-chain analysis tools. **4. Excessive Complexity:** Trying to include too many variables and constraints in a single prompt can confuse the AI and lead to an incomplete or incoherent response. It is better to split the analysis into sequential prompts.

## URL
[https://medium.com/@limingchao333/5-prompts-for-defi-investment-you-should-know-af4f1cca5770](https://medium.com/@limingchao333/5-prompts-for-defi-investment-you-should-know-af4f1cca5770)
