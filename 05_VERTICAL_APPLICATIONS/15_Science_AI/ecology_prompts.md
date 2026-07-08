# Ecology Prompts

## Description
A prompt engineering technique focused on applying Large Language Models (LLMs) to areas related to **ecology, conservation, environmental management, and sustainability**. The term "Ecology Prompts" covers the creation of instructions that guide the AI to process environmental data, generate impact reports, analyze ecological statistics, and assist in sustainable decision-making.

The concept manifests in two main strands:

1.  **Prompts for Ecological Applications:** Using AI as a tool to solve environmental problems (for example, biodiversity monitoring, environmental impact assessment - EIA).
2.  **Green Prompting:** Using prompts to reduce the energy consumption and environmental impact of the AI itself, focusing on efficiency and minimizing trial and error [2].

This technique is crucial for leveraging AI's potential in environmental research and management, ensuring that the complexity of ecological data (such as spatial and temporal structuring) is handled with the appropriate rigor [1].

## Examples
```
**1. Ecological Statistical Analysis (with CoT)**
`"Act as an ecological statistician. I have a species count dataset (dependent variable) across 5 different sites (categorical variable) over 10 years (temporal variable).
1. Which statistical test (and why) is most appropriate for analyzing the effect of site and time on species count, considering the non-independence of the data?
2. Generate the complete R code to run this test, including importing a CSV file named 'dados_ecologia.csv'.
3. Interpret the results concisely, focusing on the p-value and the effect size."`

**2. Environmental Impact Assessment (EIA)**
`"Using the US EPA 2025 guidelines for air quality, create a detailed checklist for the data collection phase of an Environmental Impact Assessment for the construction of a new highway. The checklist should include monitoring parameters, collection frequency, and recommended data sources."`

**3. Conservation Planning**
`"Identify 5 innovative site-planning strategies for creating a wildlife corridor in a fragmented urban area. Base the strategies on the 'Urban Eco 2025' report and include the ecological rationale for each one."`

**4. Regulation Interpretation**
`"Summarize the upcoming 2025 wetland protection regulations and their direct implications for the design of infrastructure projects in the state of São Paulo. Present the summary in table format with the columns: Regulation, Key Requirement, Project Implication."`

**5. Awareness Content Generation**
`"Write clear and motivating copy for a recycling awareness campaign, using an 'optimistic and action-oriented' tone of voice. The target audience is young adults (18-25 years). The text should be no more than 150 words and include a call to action (CTA)." [3]`

**6. Prompt Optimization (Green Prompting)**
`"Rewrite the following prompt to be more concise and direct, while preserving the original intent of generating a 5-point summary about photovoltaic solar energy: 'Please provide me with a very detailed and comprehensive summary, with at least 5 main points, about the benefits and challenges of implementing large-scale photovoltaic solar energy systems in developing countries.'"`

**7. Trend Identification**
`"What are the three main trends in renewable energy adoption as of 2023, according to post-2023 technology review and research documents? Structure the response with the trend name, a brief description, and the potential market impact." [3]`
```

## Best Practices
**1. Detailed Ecological Contextualization:** Always include as much domain-specific context as possible in the prompt, such as the species, the ecosystem, the geographic location, and the time period.
**2. Statistical Structure and Rigor:** When requesting statistical analyses (as in ecology), separate the workflow into components (for example, "generate R code for regression", "interpret results") and use techniques such as **Chain of Thought (CoT)** to force the AI to reason step by step, ensuring rigor [1].
**3. Output Format Specification:** Clearly define the desired format (table, checklist, Python/R code, report) so the AI can structure the response usefully.
**4. Use of Documentary References:** Mention specific documents, reports, or guidelines (for example, "Using the US EPA 2025 guidelines") to anchor the AI's response in real data and regulations [3].
**5. Green Prompting (Efficiency):** To reduce the environmental and computational cost of AI, use concise and direct prompts. Avoid overly long or ambiguous prompts that require multiple iterations or that result in long and unnecessary responses [2].
**6. Human Oversight:** Maintain human oversight over statistical decisions and environmental conclusions, since LLMs may present reasoning limitations in complex statistical tests, especially those with spatial and temporal structuring [1].

## Use Cases
**1. Ecological Research and Analysis:** Assist scientists in choosing appropriate statistical models for complex ecological data (with spatial and temporal structuring) and in generating code for data analysis (R, Python) [1].
**2. Environmental Management and Compliance:** Create checklists, summarize regulations (for example, US EPA, local laws), and generate compliance reports for infrastructure and development projects [3] [4].
**3. Environmental Impact Assessment (EIA):** Generate checklists for data collection, identify potential impacts on wildlife, and suggest mitigation strategies for construction projects [4].
**4. Biodiversity Conservation:** Automate species identification from images or sound recordings (with external tools) and generate monitoring and conservation plans [5].
**5. Communication and Awareness:** Create engaging and informative content for sustainability campaigns, corporate ESG (Environmental, Social, and Governance) reports, and educational materials [3].
**6. AI Cost Optimization and Sustainability (Green Prompting):** Apply the technique to reduce LLM energy consumption by minimizing the number of tokens and the complexity of inferences, aligning the technology with sustainability goals [2].

## Pitfalls
**1. Lack of Ecological Context:** Omitting crucial details (such as the spatial/temporal scale, the data type, or the species) leads to generic or statistically incorrect responses, especially in ecology, where data structuring is vital [1].
**2. Blind Trust in Statistical Analyses:** LLMs may generate statistical code (for example, in R or Python) that looks correct but applies the wrong test or misinterprets the results. Human oversight is indispensable for validating the choice of model and the interpretation [1].
**3. Ambiguous or Overly Long Prompts:** Poorly formulated or overly long prompts increase the number of tokens processed, raising energy consumption and computational cost (the opposite of *Green Prompting*) [2].
**4. Ignoring the Data Source:** Requesting analyses or reports without specifying the data source (for example, "Analyze the impact of pollution" without providing the dataset) results in hypothetical or unverifiable information.
**5. Not Specifying the Role (Persona):** Failing to instruct the AI to act as a "Conservation Scientist" or "Environmental Consultant" can reduce the technical quality and accuracy of the response.

## URL
[https://ecoevorxiv.org/repository/view/9493/](https://ecoevorxiv.org/repository/view/9493/)
