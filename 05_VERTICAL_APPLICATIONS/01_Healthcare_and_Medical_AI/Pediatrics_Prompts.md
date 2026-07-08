# Pediatrics Prompts

## Description
Prompt Engineering in Pediatrics is the strategic application of communication techniques with Large Language Models (LLMs) to optimize clinical and administrative practice in child care. It involves creating detailed and contextual instructions to guide the AI to generate accurate, safe outputs appropriate to the nuances of pediatric development and pathologies. The main focus is to ensure patient safety, effective communication with parents and caregivers, and support for clinical reasoning, always respecting the need for human validation and final medical judgment [1] [2].

## Examples
```
**1. Structured Initial History Taking (Role-playing + Reasoning):**
"You are a pediatrician seeing a 5-year-old child for a first appointment. List 12 essential questions for the initial history, dividing them into blocks: chief complaint, developmental history, medical background, current habits, and vaccination. Before listing the questions, explain in 3 sentences the reason for dividing them into blocks and why this is important for clinical reasoning."

**2. Communication with Parents (Tone Specification + Output Specification):**
"Take on the role of a pediatrician. Write a WhatsApp message to the parents of an 8-year-old patient diagnosed with Acute Otitis Media. The message should be concise (maximum 50 words), with an empathetic and professional tone, and should include: the diagnosis, the antibiotic prescription (Amoxicillin), and 3 warning signs for immediate return to the emergency room."

**3. Explaining a Disease to a Child (Role-playing + Customization):**
"Take on the role of a children's health educator. Explain what Asthma is to a 6-year-old child. Use simple analogies (e.g., air tubes that get tight), easy-to-understand language, and an encouraging tone. The explanation should have no more than 5 short sentences."

**4. Creating Educational Content (Output Specification + Context):**
"Create an informational leaflet for parents about introducing solid foods (BLW - Baby-Led Weaning) for 6-month-old babies. The leaflet should be formatted in 5 main topics, with clear and accessible language, and include a list of 5 safe foods to start with."

**5. Chart Summary for a Colleague (Role-playing + Output Specification):**
"You are a pediatrics resident. Summarize the chart of patient [Name, Age, Primary Diagnosis] for the incoming on-call physician. The summary should have no more than 6 lines and highlight: Chief Complaint, Management Taken, Current Medications, and Next Steps (pending exams or specialist consultations)."

**6. Medication Dose Calculation (Specific Instruction):**
"Calculate the dose of Amoxicillin for a 15 kg child with Acute Otitis Media. Use a dose of 90 mg/kg/day, divided into two administrations. Provide the result in mg per dose and the volume to be administered, considering a 250 mg/5 ml suspension."

**7. Clinical Scenario Simulation (Reasoning Prompt + Role-playing):**
"Take on the role of a residency preceptor. Create a simulated clinical case of a 3-month-old infant with fever and irritability. Include physical exam and laboratory data. At the end, ask the user to list 3 differential diagnoses and the initial management, justifying the reasoning."

**8. Developing a Follow-up Plan (Chaining + Output Specification):**
"Based on the diagnosis of ADHD in a 14-year-old adolescent, develop a multidisciplinary follow-up plan for the next 6 months. The plan should be presented in a table with 3 columns: Month, Specialist Involved (Pediatrician, Psychologist, Neurologist), and Objective of the Appointment."

**9. Translating Technical Terms (Customization + Tone Specification):**
"Translate the term 'Monosymptomatic Nocturnal Enuresis' into language for a worried grandmother. Use a reassuring tone and explain the concept in a simple way, focusing on the fact that it is a common and treatable condition."

**10. Generating Content for Social Media (Output Specification):**
"Create 3 ideas for short posts (maximum 100 characters) for the Instagram of a pediatric clinic, focusing on the importance of flu vaccination in children. Use informal and catchy language, with a focus on parents."
```

## Best Practices
**1. Be Specific and Contextualized (Role-playing and Context):** Always define the AI's role ("You are an experienced pediatrician") and provide the complete clinical context (age, weight, history, exams). Specificity is crucial for obtaining clinically relevant and accurate responses [1] [2]. **2. Ask for the Reasoning (Reasoning Prompting):** Include the instruction for the AI to articulate the reasoning process behind its response ("Explain your reasoning step by step"). This increases transparency, allows human validation, and reduces the likelihood of "hallucinations" [1] [2]. **3. Provide Examples (Few-shot Prompting):** For tasks that require a specific format or style (e.g., chart summary, message to parents), provide one or more input/output examples to calibrate the model [1]. **4. Define the Format and Tone (Output Specification and Tone Specification):** Specify the desired output format (table, list, running text) and the tone (technical for colleagues, accessible and empathetic for parents) to ensure the result is useful for the target audience [1] [2]. **5. Iterate and Validate:** The first result is not always the best. Adjust the prompt and iterate until you get the ideal response. **Never** use the AI's output without human validation by a qualified healthcare professional [2].

## Use Cases
**1. Clinical Decision Support:** Assistance in formulating differential diagnoses, suggesting treatment protocols, and calculating medication doses based on weight and age. **2. Communication with Patients and Parents:** Generation of messages, emails, or informational leaflets in accessible and empathetic language to explain complex medical conditions, treatment plans, or test results. **3. Administrative Optimization:** Chart summaries, structured transcription of appointments, and preparation of medical certificates or reports. **4. Education and Training:** Creation of simulated clinical cases for residents and students, and preparation of educational content for parents and caregivers [1] [2]. **5. Research and Review:** Searching for and summarizing up-to-date medical literature on specific pediatric pathologies.

## Pitfalls
**1. Vagueness and Generality:** Providing generic prompts without specifying the pediatric context (e.g., "calculate the dose" without the child's weight) leads to imprecise or dangerous responses [2]. **2. Bias and Hallucinations:** AI can incorporate biases from training data, or worse, generate "hallucinations" (false information presented as facts) that are unacceptable in a sensitive clinical environment such as pediatrics [3]. **3. Lack of Clinical Context:** Not including crucial information such as age, weight, developmental stage, or vaccination history in the prompt prevents the AI from considering the nuances of child care [2]. **4. Over-reliance:** Using AI as a substitute for clinical judgment or for making final decisions. AI should be a support tool, not a substitute for the healthcare professional [3]. **5. Ignoring the Need for Validation:** The AI's output, especially in dose calculations or diagnostic suggestions, must always be validated by reliable medical sources and by the responsible physician [1].

## URL
[https://www.childrenshospitals.org/news/childrens-hospitals-today/2024/12/5-tips-for-effective-ai-prompts](https://www.childrenshospitals.org/news/childrens-hospitals-today/2024/12/5-tips-for-effective-ai-prompts)
