# Accessibility Design Prompts

## Description
Accessibility Design Prompts are specific, detailed instructions provided to Large Language Models (LLMs) or generative AI tools to assist in the process of designing and developing digital products (websites, applications, systems) that are usable by people with the widest possible range of abilities and disabilities. [1] [2]

This *Prompt Engineering* technique focuses on explicitly incorporating accessibility standards (such as WCAG - Web Content Accessibility Guidelines, ADA, ATAG) and inclusive design considerations into AI requests. The goal is to leverage the AI's ability to:
*   **Compliance Analysis:** Evaluate existing or proposed designs against technical accessibility criteria.
*   **Generation of Inclusive Solutions:** Suggest design alternatives, microcopy, or user flows that meet specific needs (e.g., low vision, motor disability, cognitive disability).
*   **Documentation and Training:** Create checklists, testing protocols, and summaries of accessibility requirements. [1]

By providing context, standards, and the target audience, the designer or developer transforms the AI into an "accessibility co-pilot", accelerating the integration of inclusive practices from the earliest phases of the project. [3]

## Examples
```
**1. Contrast and Color Analysis:**
`"Act as a WCAG 2.1 AA expert. Analyze the following color palette (Primary: #007BFF, Secondary: #6C757D, Background: #FFFFFF). For each text/background color pair, calculate the contrast ratio and indicate whether it meets the requirement for normal text and large text. If it does not, suggest the closest text color that meets it."`

**2. Batch Alternative Text (Alt Text) Generation:**
`"I have a folder of 50 e-commerce product images. For each image, generate a concise and descriptive alternative text, focusing on function and visual content. The target audience is screen reader users. Example input: 'product_image_123.jpg (blue running shoe with yellow details)'. Format the output as a CSV table with 'File Name' and 'Alternative Text'."`

**3. Keyboard Navigation Optimization:**
`"Act as an accessibility tester. Describe the keyboard navigation flow for a 5-step checkout form. Identify possible focus traps or non-focusable elements. Suggest improvements to the 'tabindex' attribute and the focus order to ensure a smooth experience for users who do not use a mouse."`

**4. Microcopy Creation for Cognitive Accessibility:**
`"Write clear and empathetic microcopy for the following error messages in a banking app, focusing on users with cognitive disabilities. Use simple language (5th-grade reading level) and avoid jargon: 1. Incorrect password. 2. Insufficient balance. 3. Session expired. For each error, provide a clear and immediate solution."`

**5. Generation of Inclusive Design Guidelines:**
`"Based on the WCAG 2.2 guidelines, generate a set of 5 design rules to ensure that interactive elements (buttons and links) in a mobile app are accessible to users with motor disabilities. Include requirements for minimum touch target size and spacing between elements."`

**6. Content Structure Evaluation:**
`"Analyze the following blog article outline. Evaluate whether the heading structure (H1, H2, H3) is logical and whether the use of lists and short paragraphs maximizes readability for users with dyslexia or cognitive disabilities. Suggest a restructuring if necessary."`
```

## Best Practices
**1. Be Specific and Contextualized:** Always include the accessibility standard (e.g., WCAG 2.1 AA, ADA 2025) and the target audience (e.g., users with visual disabilities, motor disabilities) in the prompt. [1] [2]
**2. Reference Documents:** Attach or ask the AI to reference internal documents (style guides, usability test results) or external ones (WCAG, AT manuals) for more accurate responses. [1]
**3. Define the AI's Role:** Start the prompt with "Act as an inclusive design and accessibility expert" to set the tone and focus of the response. [3]
**4. Focus on Action:** Ask for actionable results, such as "Generate a checklist", "Compare compliance", or "Write microcopy".
**5. Iterate and Refine:** Use the AI's initial result as a basis and refine the prompt to deepen the analysis or shift the focus (e.g., from contrast to keyboard navigation).

## Use Cases
**1. UI/UX Optimization:** Generation of interface themes, suggestion of color palettes with approved contrast, and optimization of layouts for different needs (e.g., high contrast, enlarged text). [1]
**2. Rapid Compliance Testing:** Creation of checklists and automated testing protocols to verify adherence to standards such as WCAG 2.1 or 2.2 (levels A or AA) in early development stages. [3]
**3. Inclusive Content Generation:** Writing microcopy, error messages, and form labels that are clear, concise, and easy to understand for users with cognitive disabilities. [1]
**4. Documentation and Training:** Creation of summaries of legal requirements (e.g., ADA, Section 508) and translation of complex technical guidelines into plain language for design and development teams.
**5. Specific Interaction Design:** Suggestion of user flows for assistive technologies (e.g., keyboard navigation, voice commands) and definition of touch target sizes for users with motor disabilities. [1]

## Pitfalls
**1. Over-reliance on AI:** AI is a tool, not a substitute for manual testing and testing with real users. Compliance generated by AI should always be verified. [2]
**2. Vague Prompts:** Requests like "Make this design accessible" are too broad and result in generic, useless responses. The lack of specific standards and contexts is the main pitfall. [4]
**3. Ignoring the Human Context:** AI may fail to capture nuances of the design or usage context that affect real accessibility (e.g., the relevance of an alternative text to the context of the page).
**4. Bias in Training Data:** If the AI's training data does not include examples of truly inclusive design, the suggestions may perpetuate inaccessible design practices.
**5. Failure to Reference Standards:** Not specifying the standard (WCAG 2.1, 2.2, 3.0) or the compliance level (A, AA, AAA) can lead to results that do not meet legal or project requirements.

## URL
[https://clickup.com/p/ai-prompts/ui-ux-design-and-accessibility](https://clickup.com/p/ai-prompts/ui-ux-design-and-accessibility)
