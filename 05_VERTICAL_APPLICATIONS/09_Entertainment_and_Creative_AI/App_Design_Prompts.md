# App Design Prompts

## Description
**App Design Prompts** are structured, detailed instructions provided to generative Artificial Intelligence (AI) models (such as LLMs and image models) to assist with or automate tasks in the process of designing user interfaces (UI) and user experience (UX) for mobile and web applications. They transform conceptual ideas into tangible design assets, such as wireframes, user flows, style specifications, component code, and UX copy. The effectiveness of these prompts lies in their ability to incorporate design principles, platform guidelines, and specific user needs, acting as a virtual "design engineer". The goal is to accelerate the ideation, prototyping, and specification phase, allowing designers to focus on more complex and strategic UX problems.

## Examples
```
**1. Wireframe Generation:**
"Create a low-fidelity wireframe for the 'Checkout' screen of an e-commerce application. The goal is to minimize friction. Include the following elements: item list, delivery address field, payment options (card, Pix), and a prominent 'Complete Purchase' button. Use a 'direct and secure' tone. Output format: Markdown description and component list."

**2. User Flow:**
"Map the complete user flow for the 'First Login and Onboarding' of a meditation application. The flow should have 4 steps: 1. Welcome Screen, 2. Goal Selection (e.g., Reduce Stress), 3. Notification Permission, 4. Home Screen. For each step, suggest UX copy in a 'calm and encouraging' tone. Output format: Markdown table."

**3. Style and Accessibility Specification:**
"Suggest a color palette and typography for a personal finance management application. The palette should be based on shades of blue and green, conveying trust and growth. Ensure that all text and background color pairs meet WCAG AA accessibility standards. Typography: A modern, legible sans-serif font. Output format: A 5-color palette (HEX, RGB) and 2 fonts (Name, Weight)."

**4. Component Code Generation:**
"Generate the React Native code for a 'Notification Card' component for a news application. The card should include: a category icon, a title (max. 50 characters), a summary (max. 100 characters), and a timestamp. The design should follow Material Design guidelines. Output format: React Native code block."

**5. Error UX Copy:**
"Write the UX copy for an error message that appears when a user tries to submit a form without filling in a required field. The tone should be 'helpful and friendly', avoiding blame. The message should clearly indicate the problem and the solution. Output format: Error message text and action button text."

**6. Competitive Analysis:**
"Analyze the home screen of the 'Duolingo' and 'Babbel' apps. Identify the 3 main UI elements that promote engagement and retention. Suggest how we can adapt these elements for a new language-learning app focused on conversation. Output format: Comparative analysis in paragraphs."
```

## Best Practices
**Prompt Structure (5 C's):** An effective prompt should contain **Clarity** (what to do), **Context** (for whom and where), **Specificity** (technical and visual details), **Tone** (the brand/app voice), and **Format** (the type of desired output, such as wireframe, code, or text).
**Iteration and Refinement:** Start with simple prompts and add layers of complexity. Use the output of the first prompt as context for the next.
**Defining Constraints:** Include accessibility constraints (WCAG), platform guidelines (iOS Human Interface Guidelines, Material Design), and specific color palettes.
**Focus on the UX Problem:** Instead of just asking for a beautiful design, ask the AI to solve a user experience problem, such as "reduce cart abandonment" or "simplify the onboarding process".
**Use of Structured Data:** For outputs such as feature tables or user flows, request the output format as JSON or Markdown to facilitate integration with other tools.

## Use Cases
**Rapid Ideation and Brainstorming:** Quickly generate multiple variations of layout, color palettes, or feature concepts for the initial phase of the project.
**Wireframe and Prototype Creation:** Transform requirement specifications into low- or medium-fidelity visual sketches.
**UX Copy Generation:** Create text for buttons, error messages, notifications, and onboarding flows that align with the brand's tone of voice.
**Design System Specification:** Define and document UI components, spacing rules, typography, and accessibility for a Design System.
**Design-to-Code Translation:** Generate UI component code (e.g., React, Vue, Swift) from design descriptions, accelerating the handoff to development.
**UX Analysis and Optimization:** Ask the AI to analyze an existing user flow and suggest improvements based on usability principles.

## Pitfalls
**Vague or Generic Prompts:** Asking for "a beautiful app design" without specifying the audience, goal, or style results in irrelevant or clichéd outputs.
**Ignoring the UX Context:** Focusing only on aesthetics (UI) and neglecting the user flow, information hierarchy, and problem-solving (UX) leads to visually pleasing but dysfunctional designs.
**Over-Reliance on the First Output:** AI is an ideation tool. The first output is rarely the final solution. It is crucial to iterate, refine, and apply human design judgment.
**Copyright Infringement/Plagiarism:** Using prompts that directly imitate the style of an existing app can lead to intellectual property issues. Always seek inspiration in principles, not in direct copies.
**Lack of Technical Specifications:** Not including the framework (React Native, Flutter, Web) or the design guidelines (Material Design, iOS HIG) can generate code components or layouts that are not implementable.

## URL
[https://medium.com/@uxraspberry/prompt-engineering-for-designers-a-practical-guide-what-i-learned-so-far-140d70879c7e](https://medium.com/@uxraspberry/prompt-engineering-for-designers-a-practical-guide-what-i-learned-so-far-140d70879c7e)
