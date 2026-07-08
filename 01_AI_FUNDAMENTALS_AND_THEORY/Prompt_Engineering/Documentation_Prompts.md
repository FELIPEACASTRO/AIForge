# Documentation Prompts

## Description
"Documentation Prompts" are structured, detailed instructions provided to a large language model (LLM) with the specific goal of generating, reviewing, or improving technical documentation, user manuals, knowledge base articles, or FAQs. The technique relies on providing the LLM with the four crucial elements for creating high-quality documentation: **Persona** (the role the LLM should assume, such as "Senior Technical Writer"), **Intent** (the goal of the documentation, such as "explain the installation process"), **Scenario** (the specific context, such as "for beginner users of software X"), and **Delivery** (the output format and style, such as "in Markdown format with a friendly tone"). Effective use of this technique turns the LLM into a technical writing assistant, dramatically accelerating the content creation cycle.

## Examples
```
1. **Persona:** "Act as a Level 3 Support Engineer." **Intent:** "Create a troubleshooting guide." **Scenario:** "The user is receiving the 'HTTP 503 Service Unavailable' error when trying to access the API." **Delivery:** "Generate a knowledge base article in Markdown with a numbered list of 5 diagnosis and resolution steps, starting with checking the server status."

2. **Persona:** "You are a user-manual writer focused on simplicity." **Intent:** "Write the 'Getting Started' section." **Scenario:** "The user has just installed the 'Finanças Rápidas' app and needs to set up their first bank account." **Delivery:** "Produce a concise text of no more than 3 paragraphs, using an encouraging tone and highlighting the 'Add Account' button in bold."

3. **Persona:** "Assume the role of a software security expert." **Intent:** "Review and improve the security documentation." **Scenario:** "The following excerpt describes authentication via OAuth 2.0: [EXCERPT TO BE REVIEWED]." **Delivery:** "Rewrite the excerpt to ensure the terminology is 100% accurate and that security best practices (such as the use of short-lived tokens) are emphasized."

4. **Persona:** "Be a FAQ writer for a SaaS product." **Intent:** "Generate 10 frequently asked questions and answers." **Scenario:** "The product is a Kanban-based project management tool. The most common questions are about pricing, integrations, and user limits." **Delivery:** "Create a list of 10 FAQs, with direct and concise answers, using the brand voice (professional and helpful)."

5. **Persona:** "Act as a senior developer." **Intent:** "Document the API function `calculate_tax(amount, country_code)`." **Scenario:** "The function accepts a float and a 2-letter string and returns a float. The documentation should follow the JSDoc standard." **Delivery:** "Generate the complete documentation for the function, including a description, input parameters, return type, and a usage example in Python."

6. **Persona:** "You are a technical translator fluent in Portuguese and English." **Intent:** "Translate and localize an instruction manual." **Scenario:** "Translate the following text from English to Brazilian Portuguese, maintaining a formal tone: [TEXT IN ENGLISH]." **Delivery:** "Provide the translation in Portuguese, ensuring that technical terms such as 'firmware' and 'interface' are kept or correctly translated for the Brazilian context."
```

## Best Practices
**Be Specific and Structured:** Use the four key elements (Persona, Intent, Scenario, and Delivery) to structure your prompt. **Define the Output Format:** Specify the desired format (Markdown, HTML, JSON, manual style, etc.) and the structure (titles, subtitles, lists). **Provide Context and Examples:** Include relevant background information and, if possible, an example of existing documentation so the LLM mimics the tone and style. **Maintain Clarity and Conciseness:** Avoid ambiguity. Each instruction should be clear about what is expected. **Add Keywords:** For technical or SEO documentation, include a list of keywords to be incorporated. **Human Review is Mandatory:** Always review and edit AI-generated content to ensure technical accuracy, tone of voice, and the absence of factual errors.

## Use Cases
**User Manual Creation:** Rapid generation of step-by-step guides for new products or features. **Knowledge Base Development:** Mass production of support articles and FAQs to reduce support ticket volume. **API and Code Documentation:** Creation of standardized technical documentation (such as JSDoc, Sphinx, or OpenAPI) for functions, classes, and API endpoints. **Localization and Translation:** Translation and adaptation of existing documentation for different languages and cultural contexts. **Tutorial and Guide Generation:** Creation of educational content and tutorials for customer onboarding. **Review and Standardization:** Using the prompt to review existing documentation, ensuring consistency of tone, style, and technical accuracy.

## Pitfalls
**Lack of Context:** Not providing the Persona, Intent, Scenario, or Delivery leads to generic and useless documentation. **Excessive Dependence:** Blindly trusting the AI's output without human review can result in serious technical errors or outdated information. **Vague Prompts:** Using prompts like "Write about product X" without specifying the target audience, the goal, or the format. **Ignoring Tone of Voice:** Not defining the tone (formal, friendly, technical) results in documentation inconsistent with the brand. **Absence of Examples:** Not providing an example of existing documentation prevents the LLM from replicating the style and structure optimally. **Focusing Only on Content:** Forgetting to specify the delivery structure (titles, lists, bold) can produce a block of text that is difficult to read.

## URL
[https://betterdocs.co/ai-prompt-writing-for-documentation/](https://betterdocs.co/ai-prompt-writing-for-documentation/)
