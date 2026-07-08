# Pitch Deck Prompts

## Description
**Pitch Deck Prompts** are a specialized form of prompt engineering designed to leverage Large Language Models (LLMs) and generative AI to assist in the creation, refinement, and structuring of investor-ready *pitch decks*. Rather than generating the entire presentation, these prompts focus on producing high-quality content for specific slides (such as Problem, Solution, Market Size, Team, Investment Ask) or refining existing content for clarity, conciseness, and investor appeal. The technique relies on providing the AI with detailed context about the *startup*, its market, and its financial situation, often using a structured, multi-step approach where the AI acts as a copilot or strategic consultant [1]. The goal is to translate the founder's technical and product vision into a compelling financial and market narrative.

## Examples
```
1. **Role and Context Setup (Problem Slide):**
```
Act as an experienced Venture Capital Analyst, specialized in B2B SaaS. My startup, [Startup Name], is raising a $1.5M Seed round. Our product is [brief product description]. Generate a concise and compelling Problem Slide for our pitch deck, focusing on the pain points of [Target Customer] and the current market's failure to solve them.
```
2. **Solution and Competitive Moat (Solution Slide):**
```
Based on the Problem Slide you just generated, now create the Solution Slide. The solution should be presented as the inevitable answer to the problem. Explicitly include our competitive moat: [Mention proprietary technology, network effect, or exclusive data].
```
3. **Market Sizing (TAM/SAM/SOM):**
```
Generate the content for the Market Analysis slide. Our TAM is [X] in [Region]. We segment [Specific Segment]. Provide a clear, top-down and bottom-up calculation for our SAM and SOM, and suggest a compelling 'market wedge' strategy.
```
4. **Traction Metrics and Unit Economics (Traction Slide):**
```
Draft the content for the Traction Slide. Focus only on paid-user metrics: [Number of Paid Users], [Monthly Recurring Revenue - MRR], [Customer Churn Rate], and [LTV/CAC Ratio]. Present this data in a way that demonstrates exponential growth and strong unit economics.
```
5. **The Ask and Use of Funds (Ask Slide):**
```
We are asking for $1.5M. Detail the 'Use of Funds' slide, allocating the capital across three main categories (e.g., Product Development, Sales and Marketing, Operations) with specific percentages. The narrative should justify how this funding leads to the next major milestone (Series A).
```
6. **Language Refinement:**
```
Revise the Problem and Solution slides. Make the language more concise and impactful, reducing the word count by 30%. Ensure the tone is confident and speaks directly to the investor's perspective.
```
7. **Expertise-Focused Team Slide:**
```
Generate the content for the Team Slide. Highlight the unique expertise of our three co-founders: [Founder 1 - Expertise], [Founder 2 - Expertise], [Founder 3 - Expertise]. The focus should be on why this specific team is uniquely qualified to execute this vision.
```
```

## Best Practices
1. **Structured, Multi-Step Prompting:** Use a sequence of prompts, dedicating one or more prompts to each essential slide (Problem, Solution, Market, Finance, etc.) to ensure depth and focus.
2. **Detailed Context and Persona:** The prompt should include specific data, the target audience (e.g., Seed-stage VCs), the investment stage, and, crucially, instruct the AI to act as a "Venture Capital Analyst" or "Experienced Founder" to refine the tone.
3. **Focus on Investor Metrics:** Ask the AI to focus on metrics that matter to investors (e.g., LTV/CAC, retention, revenue growth) and avoid vanity metrics.
4. **Iterative Refinement:** Use follow-up prompts to refine the AI's output, requesting clarity, conciseness, or a different approach (e.g., "Make this slide more concise and highlight the competitive moat").
5. **Keep the Focus on the Business:** Instruct the AI to keep the content of the main slides focused on the business and strategy, moving excessive technical details to an appendix.

## Use Cases
1. **Content Generation:** Creating persuasive text for individual slides (e.g., problem statement, solution description, team biographies).
2. **Structure and Flow:** Defining the ideal sequence and content for a 10-12 slide *deck* based on the company's stage and sector.
3. **Language Refinement:** Translating technical jargon into clear, investor-friendly language.
4. **Competitive Analysis:** Generating key talking points for the "Competition" slide, highlighting the unique competitive advantage.
5. **Financial Narrative:** Crafting the narrative around financial projections and the "Investment Ask" slide (*Ask*).

## Pitfalls
1. **Excessive Technical Detail:** The AI-generated content can be too technical. The prompt should explicitly ask to keep the main slides business-focused and move the specifications to the appendix [2].
2. **Generic Content ("AI Slop"):** Relying solely on the AI without providing proprietary and specific data results in generic and unconvincing content that does not stand out to investors.
3. **Vanity Metrics:** The AI may focus on "vanity traction" (e.g., free users) if it is not instructed to prioritize paid-user, retention, and revenue metrics [2].
4. **Obscure Business Model:** The AI may fail to clearly articulate the revenue model, pricing, and margin assumptions if the prompt is vague or does not include specific financial data.
5. **Ignoring Responsible AI/Compliance:** For AI *startups*, the failure to address responsible AI and compliance in the *pitch* is seen as a *red flag* by many VCs [2].

## URL
[https://medium.com/@stunspot/one-sane-prompt-for-pitch-deck-creation-7f97905d69e3](https://medium.com/@stunspot/one-sane-prompt-for-pitch-deck-creation-7f97905d69e3)
