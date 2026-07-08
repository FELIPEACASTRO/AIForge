# Ophthalmology Prompts

## Description
Prompt Engineering in Ophthalmology refers to the art and science of formulating inputs (prompts) for Large Language Models (LLMs) with the goal of optimizing the accuracy, relevance, and usefulness of responses in the context of eye health. This technique is crucial for leveraging the potential of AI to support diagnosis, patient education, and the optimization of clinical workflows. Best practices involve the use of specific instructions, role definition (Role-Playing), the application of step-by-step reasoning techniques (Chain-of-Thought), and ensuring that the ophthalmological domain context is explicitly provided. Recent research (2023-2025) demonstrates that prompt engineering can significantly increase the accuracy of LLMs in clinical tasks, such as eye disease triage, but it requires rigorous attention to data confidentiality and to the balance between accuracy and usability.

## Examples
```
1. **Differential Diagnosis with CoT and Role-Playing:**
```
Act as an ophthalmologist specializing in the retina. A 65-year-old patient with type 2 diabetes presents with progressive central vision loss in the right eye. The fundus exam reveals microaneurysms and flame-shaped hemorrhages.
1. List the 3 most likely differential diagnoses.
2. For each diagnosis, describe the clinical reasoning (Chain-of-Thought) that leads to that conclusion.
3. Present the result in a table with the columns: Diagnosis, Reasoning, Next Diagnostic Step.
```

2. **Generation of Educational Material with Specifiers:**
```
Create a 200-word informational leaflet on "Open-Angle Glaucoma" for a patient with a primary-school education level.
1. Use simple language and avoid medical jargon.
2. Include a section on the importance of treatment adherence.
3. Present the final text formatted in short paragraphs.
```

3. **Iterative Prompt for History Taking (Powerful Prompt):**
```
You are a virtual ophthalmology triage assistant. I am the patient. I have pain and redness in my left eye.
Keep asking me questions about my symptoms, medical history, and risk factors (only one question at a time) until you have enough information to suggest whether I should seek emergency care or schedule a routine appointment.
```

4. **Creation of Multiple-Choice Questions (Training):**
```
Generate 5 medical-residency-level multiple-choice questions on the pathophysiology of Proliferative Diabetic Retinopathy.
1. Each question must have 4 options and only 1 correct answer.
2. Include the correct answer and a brief justification for each question.
```

5. **Surgical Scenario Simulation (Advanced Role-Playing):**
```
Act as an experienced ophthalmic surgeon. Describe the step-by-step phacoemulsification technique for cataract surgery in a patient with a shallow anterior chamber.
1. Highlight the 3 critical points requiring attention.
2. Use appropriate technical terms.
```

6. **Clinical Protocol Optimization (Feedback Loop):**
```
Here is our current protocol for follow-up of post-operative corneal transplant patients: [INSERT PROTOCOL HERE].
1. Analyze the protocol and suggest 3 improvements to optimize workflow and patient safety.
2. Based on your suggestions, rewrite the "Appointment Frequency" section of the protocol.
```
```

## Best Practices
**Specific Instructions:** Provide detailed and contextual instructions (e.g., "Summarize the epidemiology, pathophysiology, diagnosis, and treatment of X").
**Use of Specifiers:** Define the format and the level of detail/complexity (e.g., "Provide a brief/detailed summary", "Explain at a consultant level", "Present in a table/bullet points").
**Priming (Preparing the LLM):** Define the role and the expected input/output format (e.g., "I will provide X; I want you to return Y to me").
**Uncertainty Prompts:** Include the instruction "If you are not sure of the answer, say that you do not know" to reduce the chance of hallucinations.
**Chain-of-Thought (CoT):** Encourage the LLM to think step by step, simulating clinical reasoning to increase accuracy and transparency.
**Role-Playing:** Instruct the LLM to take on the role of an ophthalmology specialist (e.g., "Act as an experienced retina specialist.").
**Domain-Specific Context:** Ensure that the prompt includes relevant medical terminology, patient symptoms, and diagnostic criteria.

## Use Cases
**Diagnostic Assistance:** Support the triage and differential diagnosis of ophthalmological conditions (e.g., Dry Eye Disease, diabetic retinopathy).
**Patient Education:** Generation of clear educational materials adapted to the patient's level of understanding.
**Generation of Multiple-Choice Questions:** Creation of high-quality questions for the training and assessment of residents and students.
**Optimization of Clinical Workflows:** Management of the cognitive load associated with *checklists* and support in identifying medical errors.
**Analysis of Clinical Notes:** Accurate identification of ophthalmological exam components from progress notes.

## Pitfalls
**Data Confidentiality:** **NEVER** share confidential data, even de-identified data, with online LLMs.
**Memory Bias:** The LLM may be influenced by previous conversations within the same session. It is recommended to start a new session for unrelated conversations.
**Hallucinations:** Accuracy can be compromised if the prompt is not high quality, leading to incorrect or fabricated responses.
**Omission of Reasoning:** Prompts that ask the LLM to omit reasoning can result in lower-quality outputs.
**Trade-off Between Accuracy and User Satisfaction:** More complex prompts (such as CoT) can increase accuracy but also response time, affecting user satisfaction.

## URL
[https://www.nature.com/articles/s41433-023-02772-w](https://www.nature.com/articles/s41433-023-02772-w)
