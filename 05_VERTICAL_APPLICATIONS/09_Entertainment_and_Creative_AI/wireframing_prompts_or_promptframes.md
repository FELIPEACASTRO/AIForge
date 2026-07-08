# Wireframing Prompts (or Promptframes)

## Description
Wireframing Prompts, also known as **Promptframes**, represent an evolution of the traditional UX/UI wireframe, integrating generative AI prompt writing into the design process. A Promptframe is a design deliverable that documents the content goals and requirements for AI prompts, building on the layout and functionality of a wireframe. The main objective is to use AI to generate high-fidelity, relevant content to populate design elements (such as titles, button text, product descriptions, etc.), replacing the use of *lorem ipsum* and accelerating the creation of realistic prototypes for user testing. This technique sits between the creation of the low-fidelity wireframe and the detailed prototype, ensuring that content is relevant from the earliest phases of the project [1] [2].

## Examples
```
1.  **E-commerce Homepage Wireframe (High Fidelity):**
    > "Design a high-fidelity desktop wireframe for the homepage of an online running shoe store. Include a hero carousel with a clear call to action ('Shop the New Collection'), a 'Featured Products' section with 4 items, and a top navigation bar with 'Men', 'Women', 'Deals' and 'Cart'. Use a white/gray color scheme and clean typography. The layout must be responsive for mobile."
2.  **Registration Form Wireframe (Functional Focus):**
    > "Create a low-fidelity wireframe for a user registration form. The form should include fields for Full Name, Email, Password (with confirmation), and a checkbox for 'I accept the Terms of Service'. Add a 'Register' button. Include annotations for the 'Email already registered' error state and the success state after submission."
3.  **Dashboard Wireframe (SaaS):**
    > "Develop a desktop wireframe for a SaaS project management dashboard. The layout should consist of a fixed left sidebar menu (with icons for Dashboard, Tasks, Members and Settings) and a main content area. The main area should display a simplified Gantt chart and a 'Pending Tasks' list. The design should be minimalist and focused on usability."
4.  **Mobile App Wireframe (Empty State):**
    > "Generate a mobile wireframe for the 'My Recipes' screen of a cooking app. The wireframe should focus on the **empty state** of the screen. Include an illustrative icon, the text 'You haven't saved any recipes yet' and a 'Explore Recipes' call-to-action button. The design should be friendly and encouraging."
5.  **Content Block Wireframe (Layout Constraint):**
    > "Create a content block for the 'Our Values' section of a corporate website. The block should have a three-column layout, each containing an icon, a title (maximum 5 words) and a short description (maximum 2 sentences). Use a linear, modern icon style. **Constraint:** Avoid the use of vibrant colors, sticking to navy blue and white tones."
6.  **Checkout Wireframe (Flow):**
    > "Design the wireframe for a single checkout screen for an e-commerce store. The screen should consolidate the 'Shipping Information', 'Payment Method' and 'Order Summary' steps. Use a single-column layout and highlight the final total price. Include a field for a 'Discount Coupon' and a 'Complete Purchase' button."
```

## Best Practices
*   **Be Specific and Structured:** Clearly define the purpose, layout, fidelity (low or high) and functional components. The more details, the better the initial result [2].
*   **Define the Fidelity:** Specify whether the goal is a simple sketch (*low-fidelity*) or a more detailed design with typography and spacing (*high-fidelity*).
*   **Include Constraints:** Use design constraints (e.g., "Use an 8px grid", "Avoid sidebars") to guide the AI and avoid generic *templates* [2].
*   **Think About Accessibility:** Include accessibility requirements in the prompt, such as "All interactive elements must meet WCAG AA standards for contrast" [2].
*   **Iteration and Refinement:** Use the AI output as a starting point. Combine variations and manually refine the elements for alignment and usability.

## Use Cases
*   **Prototype Acceleration:** Rapid generation of prototypes populated with realistic content for immediate usability testing.
*   **Content Validation:** Test the effectiveness of different text (titles, CTAs) in a layout context before investing in high-fidelity design.
*   **Design Documentation:** Create a design artifact (*Promptframe*) that serves as a bridge between the wireframe and the prototype, communicating content requirements to *stakeholders* and developers [1].
*   **Edge-State Design:** Generation of wireframes for non-ideal states, such as empty screens, error messages and loading states.
*   **Layout Exploration:** Quickly explore multiple layout variations for the same screen, simply by changing the prompt instructions.

## Pitfalls
*   **Generic Templates:** The AI may generate conventional and predictable layouts. **Solution:** Use constraints and feed the AI examples of unconventional designs [2].
*   **Ignoring Edge States:** The AI tends to focus on the ideal flow, ignoring empty or error states. **Solution:** Explicitly request the design of these states in the prompt [2].
*   **Accessibility Problems:** The AI may neglect contrast, focus order or alt-text hints. **Solution:** Define accessibility rules as mandatory requirements in the prompt [2].
*   **Loss of Rationale:** The AI does not explain the *why* behind its design choices. **Solution:** Ask the AI to include annotations explaining the logic behind each decision.
*   **Over-Reliance:** Treating the AI output as the final design, rather than a starting point that requires manual refinement and user validation.

## URL
[https://www.nngroup.com/articles/promptframes/](https://www.nngroup.com/articles/promptframes/)
