# Patent Application Prompts

## Description
Patent Application Prompts refer to the strategic formulation of instructions and text inputs for Large Language Models (LLMs) to assist in drafting patent documents. The goal is to guide the AI to generate specific sections of the application, such as the Detailed Description, the Claims, the Abstract, and the Background of the Invention, ensuring that the text is technically accurate, legally sound, and compliant with the requirements of regulatory bodies such as the INPI (Brazil) or USPTO (USA). The technique focuses on providing detailed context about the invention, defining the AI's role (for example, as a junior patent attorney), and applying format and legal style constraints to optimize the efficiency and quality of the initial draft. It is a productivity-enhancement tool for Intellectual Property professionals.

## Examples
```
**1. Independent Claim Draft (Method):**
```
Act as a senior patent attorney. Based on the description of the invention provided below, draft Claim 1 (independent) for a method. The claim should be written in the 'preamble + comprising the steps of:' format and focus only on the essential elements that define the novelty.

[DESCRIPTION OF THE INVENTION]: A new method for purifying water using graphene nanoparticles functionalized with a silver compound, where the steps include: 1) Synthesis of the nanoparticles, 2) Functionalization with silver in aqueous solution, 3) Addition of the nanoparticle solution to the contaminated water, 4) Exposure to UV light for 10 minutes, 5) Filtration to remove the nanoparticles.
```

**2. Background of the Invention Generation (Background):**
```
Draft the 'Background of the Invention' section for a patent application on [TITLE OF THE INVENTION]. The text should be no more than 400 words and address the current state of the art, the unresolved problems, and the limitations of existing solutions. Use a formal and technical tone.

[TITLE OF THE INVENTION]: Logistics Route Optimization System based on Reinforcement Learning.
```

**3. Detailed Specification Expansion:**
```
Expand the following excerpt of the technical specification to include implementation details, materials, and design variations. The expanded text should be didactic and provide support for the claims. Maintain a third-person, passive-voice writing style.

[ORIGINAL EXCERPT]: The processing module (101) is configured to receive data from sensors (102) and execute a filtering algorithm.
```

**4. Abstract Generation (Abstract):**
```
Based on the set of claims and the description of the invention, write a concise Abstract (maximum 150 words) for the patent application. The abstract should describe the essence of the invention, the problem solved, and the main element of novelty.
```

**5. Clarity and Legal Ambiguity Review:**
```
Analyze the following claim. Identify and suggest revisions for any ambiguous language, vague terms, or phrases that could be interpreted too broadly or too narrowly, compromising legal clarity.

[CLAIM]: A device for improving energy efficiency, comprising a control component that intelligently adjusts power.
```

**6. Generating Headings and Subheadings for the Specification:**
```
Generate a structure of headings and subheadings (Table of Contents) for the 'Detailed Description of the Invention' section of a patent on [SUBJECT]. Include standard sections such as 'Brief Description of the Drawings' and 'Implementation Examples'.

[SUBJECT]: Wearable Device for Continuous Non-Invasive Glucose Monitoring.
```

**7. Dependent Claim Draft:**
```
Draft three dependent claims (Claims 2, 3, and 4) that narrow Claim 1 provided below. The restrictions should focus on (a) a specific material, (b) a numerical parameter range, and (c) an optional additional step.

[CLAIM 1]: A method for purifying water, comprising the steps of: adding functionalized nanoparticles to a contaminated water source; and filtering the water to remove the nanoparticles.
```
```

## Best Practices
**1. Separate Instructions from Context:** Use clear separators (such as `###` or `---`) to distinguish the prompt instructions from the input text (the invention context). This ensures that the AI model understands what is a command and what is data. **2. Be Specific and Detailed:** Provide precise instructions about the desired result, including context, format, style, and word/paragraph limits. Avoid vague descriptions such as "short" or "several sentences." **3. Use State-of-the-Art Models:** Prioritize more advanced models (such as GPT-4 or higher) for complex patent-drafting tasks, as they are more reliable, creative, and capable of handling more nuanced instructions. **4. Adjust the Creativity Level (Temperature):** Keep the 'temperature' level (creativity/randomness) low (close to 0) for most technical and legal sections, such as claims and specifications, to reduce the risk of 'hallucinations' and ensure accuracy. **5. Chain Prompts (Fine-Tuning):** Use the output of one prompt as the input for a subsequent prompt (chaining) to refine the text. For example, generate a draft and then use a second prompt to revise conciseness or legal tone.

## Use Cases
**1. Initial Draft of Sections:** Rapid generation of initial drafts for less critical or more standardized sections of the application, such as the Background of the Invention and the Abstract. **2. Specification Expansion:** Detailing concepts and technical elements to ensure that the claims have sufficient descriptive support (enablement). **3. Dependent Claim Generation:** Creating a set of dependent claims that narrow the main claim, exploring different scopes of protection. **4. Terminology Translation and Adaptation:** Translating technical documents or adapting terminology to the required patent legal jargon (for example, transforming engineering language into patent language). **5. Prior Art Analysis:** Summarizing and analyzing existing patent documents (prior art) to identify gaps and define the novelty of the invention. **6. Style and Format Review:** Reviewing drafts to ensure style consistency, grammar, and adherence to specific numbering and drawing-reference formats. **7. Creating Questions for the Inventor:** Generating a list of detailed questions for the inventor to fill technical information gaps before the final drafting.

## Pitfalls
**1. Hallucinations and Technical Inaccuracy:** The AI may generate incorrect or inconsistent technical details (hallucinations), which is fatal in legal documents such as patents. **2. Breach of Confidentiality:** Using public LLMs with confidential invention data can violate non-disclosure agreements (NDAs) and compromise the novelty of the invention. **3. Vague Language or Legal Ambiguity:** The AI may use generic or ambiguous language that does not meet the standard of clarity and precision required by patent laws (for example, the 'descriptive sufficiency' requirement). **4. Failure to Meet Formal Requirements:** The AI may not strictly adhere to the specific format and structure requirements of each patent office (INPI, USPTO, EPO), requiring extensive manual review. **5. Lack of Support for Claims:** Generating claims without ensuring that each element is explicitly supported and described in the specification (Detailed Description), which can lead to rejection. **6. Inventorship Issues:** The use of AI raises complex questions about who is the legal inventor, especially in jurisdictions that do not recognize AI as an inventor. **7. Overreliance:** Blindly trusting the AI's draft without a thorough technical and legal review by a qualified professional.

## URL
[https://www.patentclaimmaster.com/blog/best-practices-for-gpt-prompt-engineering-when-patent-drafting/](https://www.patentclaimmaster.com/blog/best-practices-for-gpt-prompt-engineering-when-patent-drafting/)
