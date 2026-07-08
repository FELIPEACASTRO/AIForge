# Astrophysics Prompts

## Description
The use of Large Language Models (LLMs) in astrophysics and space science involves creating specialized prompts to leverage the models' capabilities for tasks such as data analysis, theoretical modeling, educational content generation, and scientific communication. This technique is part of the broader field of "Astrocomputing" in the era of LLMs. Effective prompts frequently assign a specific persona (for example, "experienced astrophysicist") to guide the level of detail and the style of the response, and they use advanced techniques such as Chain-of-Thought (CoT) or Few-Shot Learning to handle complex scientific reasoning and data interpretation. The goal is to transform vast astronomical datasets and complex theories into actionable *insights* or accessible explanations. Success depends on the prompt's precision in defining the context, the model's role, and the expected output format.

## Examples
```
**1. Data Analysis and FITS Interpretation:**
`Act as an astronomical data scientist. Analyze the following FITS header and summarize the main observational parameters (e.g., telescope, date, exposure, object). Then, suggest three potential observational biases. [INSERT FITS HEADER DATA]`

**2. Theoretical Modeling and Simulation:**
`Act as a theoretical astrophysicist. Explain the process of stellar nucleosynthesis in low-mass stars (such as the Sun). Then, provide the Python code to simulate the luminosity curve of a Type Ia supernova, using the Astropy library.`

**3. Educational Content Generation:**
`Act as a space science educator. Create a prompt for an image generator that visualizes the accretion of matter around a stellar-mass black hole, with an X-ray-visible accretion disk. The tone should be didactic and visually impactful.`

**4. Literature Review and Synthesis:**
`Act as a literature reviewer. Synthesize the main arguments and findings from the last 5 years on the Hubble Constant controversy, citing the primary sources (if possible) and highlighting the difference between distance measurement methods. Use an executive summary format.`

**5. Astrophysics Problem Solving:**
`Act as an experienced astrophysicist. Calculate the distance of a globular cluster whose average apparent magnitude of its RR Lyrae stars is m=18.5, given that the average absolute magnitude is M=0.6. Show the calculation step by step (Chain-of-Thought) and express the result in parsecs and light-years.`

**6. Prompt for Exoplanet Research:**
`Act as an exoplanet researcher. Given the mass of a star (0.8 M☉) and the orbital period of an exoplanet (15 days), calculate the exoplanet's orbital radius in Astronomical Units (AU). Assume a circular orbit and use Kepler's Third Law. Explain the relevance of this radius to the habitable zone.`

**7. Creating a Script for Science Communication:**
`Act as a science documentary screenwriter. Create a 3-minute script for a video about the formation of the Milky Way, focusing on the Big Bang theory and dark matter. The script should be engaging, accessible to a lay audience, and include suggestions for archive footage (e.g., Hubble, JWST).`
```

## Best Practices
**1. Role Assignment:** Always begin the prompt by assigning a specific, experienced role to the LLM, such as "Act as an astrophysicist specialized in black holes" or "Act as an astronomical data scientist." This guides the tone, level of detail, and technical accuracy of the response.
**2. Data Contextualization:** When analyzing data (such as FITS, CSV, or tables), include a representative excerpt of the data or the header, and specify the desired output format (e.g., JSON, Python Pandas DataFrame).
**3. Use of Advanced Techniques:** For complex scientific reasoning or modeling problems, use techniques such as **Chain-of-Thought (CoT)**, asking the LLM to "think step by step" before giving the final answer, or **Few-Shot Learning**, providing examples of problems and solutions.
**4. Specificity and Limitation:** Be as specific as possible about the phenomenon, theory, or astronomical object. Limit the scope of the response to avoid imprecise generalizations.
**5. Cross-Verification:** Always treat the LLM's output as a starting point or a hypothesis. The complexity and critical nature of astrophysical data require **cross-verification** with primary sources, simulations, and dedicated scientific software.

## Use Cases
**1. Preliminary Data Analysis:** Assist astrophysicists in the rapid interpretation of FITS headers, observation logs, or simulation results, identifying key parameters and potential anomalies.
**2. Hypothesis Generation:** Use the LLM to quickly explore the implications of new discoveries or data, generating testable hypotheses for research.
**3. Education and Science Communication:** Create educational materials, summaries of complex articles, or video scripts, translating complex astrophysical concepts into language accessible to different audiences.
**4. Code Review and Documentation:** Generate documentation for simulation code (e.g., Fortran, Python) or review code snippets for optimization and *bug* fixing in data analysis routines.
**5. Literature Synthesis:** Perform the mining and synthesis of large volumes of scientific articles (if the LLM has access to such a database), identifying trends and gaps in current research.
**6. Conceptual Modeling:** Assist in formulating conceptual models for astrophysical phenomena, such as galaxy evolution or the dynamics of stellar systems, before starting computationally intensive simulations.

## Pitfalls
**1. Scientific Hallucinations:** The LLM may generate factually incorrect information or obsolete theories. **Pitfall:** Blindly trusting the output without cross-verification.
**2. Lack of Data Context:** Providing raw data without specifying the format, units, or observational context. **Pitfall:** The LLM may misinterpret the values or apply incorrect formulas.
**3. Vague Prompts:** Asking "tell me about black holes" without a clear objective. **Pitfall:** Receiving a generic, useless response for specialized research or education.
**4. Calculation Limitations:** Although LLMs can perform calculations, they are not calculators. **Pitfall:** Using the LLM for complex calculations that require high precision and are better performed by scientific software (e.g., NumPy, Astropy).
**5. Unit Confusion:** Astrophysics deals with complex units (parsecs, light-years, magnitudes, etc.). **Pitfall:** Failing to specify input and output units, leading to order-of-magnitude errors.

## URL
[https://www.nature.com/articles/s41597-025-04613-9](https://www.nature.com/articles/s41597-025-04613-9)
