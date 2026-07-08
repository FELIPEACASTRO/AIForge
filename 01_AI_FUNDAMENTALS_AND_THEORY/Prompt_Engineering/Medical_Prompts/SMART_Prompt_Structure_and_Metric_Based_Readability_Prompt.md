# SMART Prompt Structure (Seeker, Mission, AI Role, Register, Targeted Question) and Metric-Based Prompt for Readability

## Description

A structured prompt engineering technique developed to improve the quality, accuracy, and relevance of LLM responses in healthcare settings, particularly for patient questions and the generation of educational materials. The SMART acronym represents the five essential components that guide the LLM toward an optimized response. A 2025 study (Vaira et al.) demonstrated that the SMART format significantly improved the accuracy, clarity, relevance, completeness, and overall usefulness of GPT-4o responses in head and neck surgery, compared to unstructured prompts. A prompt engineering technique that incorporates explicit readability constraints and linguistic metrics (such as 6th-grade reading level, short sentences, and simple words) to force the LLM to generate patient education materials that meet health literacy standards. This approach is more effective than prompts that merely request 'plain language'. A 2025 study (Ellison et al.) in colorectal surgery demonstrated that the Metric-Based Prompt consistently produced the most readable content. ChatGPT, using this prompt, generated materials with a Reading Level of 5.2, significantly better than the average level of 8.1 of existing educational materials.

## Statistics

A 2025 study (Vaira et al.) demonstrated that the SMART format significantly improved the accuracy, clarity, relevance, completeness, and overall usefulness of GPT-4o responses in head and neck surgery, compared to unstructured prompts. A 2025 study (Ellison et al.) in colorectal surgery demonstrated that the Metric-Based Prompt consistently produced the most readable content. ChatGPT, using this prompt, generated materials with a Reading Level of 5.2, significantly better than the average level of 8.1 of existing educational materials.

## Features

Defines the user's role (Seeker), the goal of the query (Mission), the role the AI should assume (AI Role), the tone and language style (Register), and the specific question (Targeted Question). Ensures that the output is tailored to the patient's level of knowledge. Uses quantifiable metrics (for example, Flesch-Kincaid Grade Level, SMOG) directly in the prompt. Ensures content accessibility for patients with low health literacy, a key requirement for treatment adherence.

## Use Cases

Generation of accurate and understandable responses to patient questions, creation of patient education materials tailored to health literacy level, optimization of health chatbots for end-user interactions. Generation of information leaflets, discharge instructions, and consent materials that meet public health readability guidelines (generally 6th to 8th grade level).

## Integration

Example Prompt for Patient Education:\n\n**Seeker:** I am a patient seeking information about my recent diagnosis of a thyroid nodule.\n**Mission:** I want to understand what a thyroid nodule is, the possible causes, and the treatment options.\n**AI Role:** You are a medical specialist in endocrinology, providing clear and understandable health information for patients.\n**Register:** Use simple and accessible language, suitable for a patient without a medical background, and include references to reliable health sources.\n**Targeted Question:** What should I know about thyroid nodules, their causes, and treatment options? Example Metric-Based Prompt (adapted):\n\n**Instruction:** Generate patient education material about [Medical Condition].\n**Constraints:** The text MUST have a Flesch-Kincaid Reading Level of 6th grade or lower. Use short sentences (maximum of 15 words) and avoid polysyllabic words. The tone should be encouraging and informative.\n**Question:** Explain [Medical Condition] and what the patient can expect during treatment.

## URL

https://aao-hnsfjournals.onlinelibrary.wiley.com/doi/full/10.1002/oto2.70075; https://www.sciencedirect.com/science/article/abs/pii/S0039606024010110
