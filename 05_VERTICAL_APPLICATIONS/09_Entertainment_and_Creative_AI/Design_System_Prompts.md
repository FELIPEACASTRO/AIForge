# Design System Prompts

## Description
"Design System Prompts" is a prompt engineering technique that uses large language models (LLMs) and other generative AIs to create, maintain, document, or apply a **Design System (DS)**. The central concept is to provide the Design System (including design tokens, components, and brand guidelines) as **context** or a **guardrail** for the AI. This ensures that the AI's output (whether code, design, or documentation) is inherently consistent with the established design and brand standards. The recent evolution (2024-2025) focuses on integrating Design Systems as structured "knowledge" into app-building tools and AI-powered code editors, often using protocols such as the Model Context Protocol (MCP) to pass the context in a machine-readable way.

## Examples
```
1. **Component Generation (High Level):**
`"Create a 'Notification Card' component for Figma. It should use the color token 'color-brand-primary' for the header, the typography token 'font-body-medium' for the body, and the 'Button-Primary' component for the action. The structure should be: Icon (left), Title, Body Text, and Action Button (right)."`

2. **Layout Generation (With DS as Context):**
`"Using only the components available in the Design System [DS Name], generate the React code for a 'Profile Settings' page. The layout should include an 'Avatar' component, three 'Input-Text' fields for Name, Email, and Password, and a 'Button-Primary' to save. Apply the 'spacing-large' spacing between the elements."`

3. **DS Creation (Progressive Fidelity):**
`"I am creating a task management application. Create a low-fidelity Design System, asking me about the main color palette, the font family, and the three most important UI components. Ask me one question at a time."`

4. **Consistency Audit:**
`"Analyze the following HTML/CSS code snippet. Identify all instances where colors or font sizes do not match the tokens defined in our Design System (tokens: color-text-default: #333, font-size-body: 16px). Suggest the correction using the tokens."`

5. **Component Documentation:**
`"Generate the usage documentation (in Markdown) for the 'Confirmation Modal' component. Include the description, usage examples (with sample code), and a list of all 'props' (title, message, onConfirm, onCancel) with their types and default values."`

6. **Design Token Generation:**
`"Create a design tokens JSON file for a brand focused on sustainability. Define tokens for primary colors (dark green, light green), secondary colors (beige, white), typography (a serif font for headings, a sans-serif for body), and spacing (small, medium, large)."`

7. **Code Refactoring:**
`"Refactor the CSS code below to replace all hexadecimal values with the corresponding design tokens from our Design System. If a token does not exist, use the closest token and flag the change. Code to be refactored: \`background-color: #007bff; padding: 20px;\`"`

8. **Accessibility Check:**
`"Analyze the 'Button-Secondary' component of our Design System. Check whether the combination of background color ('color-background-secondary') and text color ('color-text-on-secondary') meets the minimum WCAG AA contrast (4.5:1). If it does not, suggest the closest text color token that meets it."`
```

## Best Practices
**Provide the DS as Context:** Instead of just describing what you want, provide the complete Design System (tokens, components, guidelines) as part of the knowledge context of the AI tool. **Use Progressive Fidelity (Low-to-High Fidelity):** Start with prompts to generate a low-fidelity DS (basic structure, primary colors, typography). Review and refine, and then use prompts to evolve to medium and high fidelity, adding details and complexity. **Clear Prompt Structure:** The prompt should be clear about the **objective**, the **DS context** (if it is not provided automatically), and the desired **output format** (code, description, Figma component, etc.). **Use MCP (Model Context Protocol):** For code/design tools that support MCP (such as the Figma Dev Mode MCP Server), use this integration to pass the DS context in a structured, machine-readable way, rather than relying on text alone. **Focus on "Guardrails":** Use the DS to act as "guardrails" for the AI, limiting creative choices to pre-approved elements, ensuring consistency and quality.

## Use Cases
**Consistent Component Generation:** Generate new UI components that automatically adhere to the DS's design tokens and naming conventions. **Screen/Layout Creation:** Generate entire screen layouts or interactive prototypes using only existing DS components. **Automated Documentation:** Generate technical and usage documentation for new DS components. **Refactoring and Migration:** Use the AI to refactor legacy code so that it uses DS components and tokens. **Consistency Audit:** Ask the AI to audit an existing design or code and identify violations of the DS guidelines.

## Pitfalls
**Over-Reliance on the AI's "Truth":** Assuming that the AI's output is 100% accurate and consistent with the DS without human review. **Ambiguous Prompts:** Using vague language that allows the AI to make assumptions that violate the DS guidelines. **Insufficient Context:** Not providing enough context about the DS, resulting in generic or inconsistent outputs. **Text-Only Focus:** Trying to describe a complex DS with text alone instead of using structured integrations (such as MCP) or configuration files (such as JSON tokens).

## URL
[https://www.youngleaders.tech/p/how-to-prompt-create-a-design-system-](https://www.youngleaders.tech/p/how-to-prompt-create-a-design-system-)
