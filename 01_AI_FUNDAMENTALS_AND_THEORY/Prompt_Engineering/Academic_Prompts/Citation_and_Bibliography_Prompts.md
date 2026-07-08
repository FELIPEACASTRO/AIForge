# Citation & Bibliography Prompts

## Description
The **Citation & Bibliography Prompts** technique is a Prompt Engineering strategy focused on instructing Large Language Models (LLMs) to generate bibliographic references and in-text citations with high accuracy and in specific formats (such as APA, MLA, Chicago, ABNT). The core principle is the application of the **RTCF (Role, Task, Context, Format)** method, where the user provides the LLM with a specialized role (e.g., librarian), a clear task (e.g., generate a citation), the complete **Context** (source data such as author, title, year, DOI), and the exact output **Format** (e.g., APA 7th Edition, reference list entry). By providing complete, structured context, the user mitigates the LLM's tendency to "hallucinate" source data, ensuring accuracy and compliance with academic standards. It is an essential technique for producing reliable academic and technical content.

## Examples
```
**1. Journal Article Citation (APA 7th Edition - Zero-Shot)**
```
**Role:** Act as a librarian specialized in APA 7th Edition.
**Task:** Generate the complete reference list entry for the following journal article.
**Context:**
- Authors: Smith, J. A., & Jones, B. C.
- Publication Year: 2024
- Article Title: The Future of Prompt Engineering in LLMs
- Journal Title: Journal of AI Research
- Volume: 15
- Issue: 2
- Pages: 112-130
- DOI: 10.1000/jair.2024.15.2.112
**Format:** Provide only the formatted citation.
```

**2. Book Citation (MLA 9th Edition - Zero-Shot)**
```
**Role:** You are an academic research assistant.
**Task:** Create a "Works Cited" entry for the book below.
**Context:**
- Author: Johnson, Emily
- Book Title: The Algorithmic Muse
- Publisher: Tech Press
- Publication Year: 2023
- City of Publication: New York
**Format:** Format in MLA 9th Edition.
```

**3. In-Text Citation (ABNT NBR 6023 - Direct Quotation)**
```
**Role:** Act as a Brazilian academic text editor.
**Task:** Generate the direct quotation (with quotation marks and page) in the body of the text for the following sentence, using the author-date system.
**Sentence:** "Artificial intelligence will transform higher education."
**Context:**
- Author: Silva, M. R.
- Year: 2025
- Page: 45
**Format:** ABNT NBR 6023 (Last name, Year, p. X).
```

**4. Website Citation (Chicago 17th Edition - Notes and Bibliography)**
```
**Role:** You are a Chicago style specialist.
**Task:** Generate the footnote and the bibliography entry for the web page.
**Context:**
- Author: The Prompting Guide Team
- Page Title: Advanced Prompting Techniques
- Site Name: PromptingGuide.ai
- Publication Date: October 15, 2023
- URL: https://www.promptingguide.ai/techniques/advanced
- Access Date: November 8, 2025
**Format:** Generate the complete footnote and the bibliography entry separately.
```

**5. BibTeX Generation for an Article (Technical Format)**
```
**Role:** Act as a metadata generator for LaTeX.
**Task:** Convert the source data into a BibTeX entry formatted as @article.
**Context:**
- Author: Chen, H., & Li, W.
- Title: Large Language Models as Citation Generators
- Journal: AI Review
- Year: 2024
- Volume: 8
- Pages: 50-65
**Format:** Generate the complete BibTeX code.
```

**6. Few-Shot for a Custom Style**
```
**Role:** You are a reference formatter.
**Task:** Format the Context source in Style X.
**Example (Few-Shot):**
- Example Source: Author: Adams, S. | Title: The Guide | Year: 2020
- Example Output: ADAMS, S. (2020). The Guide.
**Context:**
- Author: Baker, L.
- Title: Prompting for Success
- Year: 2025
**Format:** Format the Context source in the same Style X as the Example.
```
```

## Best Practices
**1. Provide Complete and Structured Context (RTCF):** Use the **Role, Task, Context, Format** framework. Context is the most critical; provide all source metadata (author, title, year, publisher, DOI, URL) in a clear and organized way, rather than just a link.
**2. Specify the Style and Edition:** Be explicit about the citation style (e.g., APA, MLA, ABNT) and the edition (e.g., 7th Edition, 9th Edition).
**3. Use Few-Shot Prompting for Complex Formats:** For less common styles or complex source formats (e.g., patents, government reports), include one or two examples of correct citations in the desired style before requesting the new citation.
**4. Define the Output Type:** Specify whether you need the **reference list entry** (complete bibliography) or only the **in-text citation** (parenthetical or narrative citation).
**5. Cross-Validation:** Always check the LLM's output against a traditional citation generator (such as Citation Machine or Scribbr) or against the official style manual, especially for critical academic work.

## Use Cases
**1. Academic Production:** Fast and accurate generation of reference lists (bibliographies) and in-text citations for articles, theses, dissertations, and school assignments in any style (APA, MLA, Chicago, ABNT, Vancouver).
**2. Review and Standardization:** Converting reference lists from one style to another (e.g., from MLA to APA) or standardizing the metadata of collected sources.
**3. Research and Development (R&D):** Creating BibTeX or RIS entries for reference managers (such as Zotero or Mendeley), facilitating the organization of large volumes of literature.
**4. Journalism and Technical Content:** Ensuring that all sources in a news article or technical manual are referenced consistently and professionally.
**5. Education:** A learning tool for students to understand the structure and requirements of different citation styles, using the LLM as a format checker.

## Pitfalls
**1. Relying on Links:** The most common mistake is providing only a URL and expecting the LLM to extract all metadata correctly. The LLM may "hallucinate" the author, date, or title.
**2. Lack of Specificity in the Format:** Requesting only "an APA citation" without specifying the edition (e.g., 6th vs. 7th) or the entry type (reference vs. in-text) leads to inconsistent results.
**3. Ignoring Context:** Failing to provide crucial metadata (such as DOI, edition number, or publisher name) forces the LLM to guess, increasing the error rate.
**4. Zero-Shot Prompting for Rare Styles:** For less common citation styles or very specific source formats (e.g., a university's guidelines), the LLM may fail without an example (Few-Shot) to guide the formatting.
**5. Not Citing the AI:** Forgetting to cite the LLM itself (such as ChatGPT or Gemini) when it is used to generate or analyze content, which is a growing requirement in many academic guidelines.

## URL
[https://www.getpassionfruit.com/blog/blog-ai-prompt-engineering-citations](https://www.getpassionfruit.com/blog/blog-ai-prompt-engineering-citations)
