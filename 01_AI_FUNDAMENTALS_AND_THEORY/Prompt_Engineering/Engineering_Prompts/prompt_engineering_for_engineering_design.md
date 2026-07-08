# Prompt Engineering for Engineering Design

## Description

Prompt Engineering for Engineering Design is the systematic application of prompt engineering techniques to optimize interaction with Large Language Models (LLMs) and Generative AI in the context of the design and engineering process. The goal is to extract precise, technically valid, and creative outputs for tasks such as creating Product Design Specifications (PDS), morphological charts, failure analysis, simulations (CFD/FEA), and technical documentation. Its effectiveness lies in the ability to assign engineer personas, provide detailed context, and require step-by-step reasoning (Chain-of-Thought) to ensure technical accuracy and the relevance of the proposed solutions. Recent research (2024-2025) points to the critical need for human verification of AI outputs, despite their convincing appearance.

## Statistics

**Benchmarking:** The ENGDESIGN benchmark (2025) was proposed to evaluate the ability of LLMs in practical design tasks. **Performance Metrics:** For control engineering tasks, metrics such as rise time, settling time, overshoot, and steady-state error are used. **Synthesis Performance:** Research (2025) indicates that LLMs retrieve explicit information well, but performance drops on tasks requiring complex synthesis and reasoning, such as the design process. **Reliability:** Verifying the technical accuracy of AI NLP outputs is considered critical (Design Society, 2024).

## Features

**Essential Techniques:** Persona Assignment (e.g., "Senior Mechanical Engineer"), Chain-of-Thought (CoT) Reasoning for complex analysis, Few-Shot Learning with project examples, and Constraint Specification (cost, material, laws of physics). **Applications:** Accelerating Research and Development (R&D), optimizing design processes, generating technical documentation (reports, assembly instructions), and initial validation of product ideas (Product-Market-Fit). **Resources:** Use of specific benchmarks such as ENGDESIGN to evaluate the ability of LLMs in design tasks.

## Use Cases

**Product Design:** Creating Product Design Specifications (PDS) and morphological charts. **Mechanical Engineering:** Generating Finite Element Analysis (FEA) and Computational Fluid Dynamics (CFD) reports, component selection, and creating assembly instructions. **Innovation:** Suggesting innovative designs and optimizing systems (e.g., reducing vibration in rotating shafts). **Validation:** Assessing product-market fit (Product-Market-Fit) for new hardware devices.

## Integration

**Recommended Prompt Structure:**
1.  **Persona:** "You are a senior product engineer specializing in [Area]."
2.  **Task:** "Design a [Component/System] that meets [Function]."
3.  **Context and Constraints:** "The material must be [Material], the manufacturing cost cannot exceed [Value], and it must operate under [Environmental Conditions]."
4.  **Format:** "Provide the output in list format with the following sections: 1. Design Specifications, 2. Materials Analysis, 3. Solution Sketch. Use Chain-of-Thought reasoning to justify the material choice."

**Example Prompt (Simulation Analysis):**
"You are a computational fluid dynamics engineer. Create a Computational Fluid Dynamics (CFD) report for a NACA 0012 airfoil at an angle of attack of 5 degrees and a velocity of 50 m/s. Include the mesh, the boundary conditions, and the pressure and velocity results. Use CoT reasoning to explain the discretization methodology."

## URL

https://www.designsociety.org/download-publication/47276/prompt_engineering_on_the_engineering_design_process
