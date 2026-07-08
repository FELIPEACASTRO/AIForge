# Physical Therapy Prompts

## Description
Prompt Engineering for Physical Therapy (Physical Therapy Prompts) refers to the art and science of creating text inputs optimized for Large Language Models (LLMs) and AI assistants, with the goal of obtaining clinically relevant, accurate, and actionable results in the context of physical therapy practice [1].

This technique allows physical therapists to use AI as a clinical and administrative assistant, helping with tasks such as:
1.  **Treatment Plan Generation:** Creation of personalized, phase-based rehabilitation programs [2].
2.  **Clinical Documentation:** Preparation of SOAP notes, discharge summaries, and progress reports [3].
3.  **Patient Education:** Development of educational materials and answers to common patient questions [1].
4.  **Clinical Decision Support:** Identification of prognostic factors, diagnostic tests, and contraindications [2].

The effectiveness of the prompt depends directly on the **clarity, specificity, and contextualization** of the request, with the **Role-Goal-Instruction (RGI)** structure being one of the most recommended methodologies for maximizing the clinical usefulness of the AI's response [1].

## Examples
```
**1. Creating a Rehabilitation Program (RGI)**
*   **Prompt:** "You are a physical therapist specializing in knee rehabilitation. Your goal is to create a home exercise program for a 55-year-old sedentary patient, 6 weeks after a total knee arthroplasty. The program should focus on gaining range of motion (ROM) and initial strengthening. Present the plan in a table with 3 columns: Exercise, Frequency (sets x reps), and Precautions. Include 5 exercises."

**2. Analysis of Prognostic Factors**
*   **Prompt:** "What are the 5 main prognostic factors (positive and negative) for the success of conservative treatment of nonspecific chronic low back pain, according to the 2023-2025 clinical guidelines? List them and provide a brief evidence-based justification for each one."

**3. SOAP Note Generation (Structured Output)**
*   **Prompt:** "Act as a clinical documentation assistant. Generate a SOAP note for the following scenario: Patient: 22-year-old athlete, grade II ankle sprain 48h ago. Subjective: Reports pain 7/10 (VAS) when walking and swelling. Objective: Moderate edema, positive anterior drawer test, dorsiflexion ROM limited by 50%. Assessment: Acute lateral ankle sprain. Plan: Begin gentle mobilization, isometric exercises, and cryotherapy. Format the output strictly in the SOAP format, with headings in bold."

**4. Development of Educational Material**
*   **Prompt:** "Create a one-page educational leaflet, with simple and accessible language (6th-grade reading level), for a patient with a recent diagnosis of Carpal Tunnel Syndrome. The leaflet should explain the condition, list 3 workplace ergonomics tips and 3 neural gliding exercises. Use headings and bullet points."

**5. Identification of Contraindications**
*   **Prompt:** "What are the absolute and relative contraindications for the use of shockwave therapy (ESWT) in a patient with Achilles tendinopathy? Present the answer in two separate lists and cite the source of the clinical guideline (e.g., NICE, Cochrane) that supports the information."
```

## Best Practices
**Be Specific and Contextual:** Use the **Role-Goal-Instruction (RGI)** structure to provide clear context (e.g., "You are a physical therapist specializing in sports rehabilitation"). Include relevant clinical details, such as age, activity level, time since injury, and previous treatments [1].

**Request Evidence and Format:** Explicitly ask the AI to base the response on **evidence** (e.g., "based on the 2024 APTA guidelines") and define the output format (e.g., "present in a table format with 3 columns: Exercise, Repetitions, Objective") [2].

**Iterative Refinement:** Start with a broader prompt and refine it with follow-up questions to obtain more specific details (e.g., "Expand exercise X, detailing the progression for weeks 4-6") [1].

**Maintain Confidentiality:** **NEVER** include patient identifying information (PHI) in prompts. Use anonymized and generic clinical data [1].

## Use Cases
**Clinical Decision Support:**
*   Suggestion of orthopedic tests and their sensitivity/specificity for specific conditions [2].
*   Analysis of research articles to summarize the efficacy of a new treatment modality.

**Documentation Optimization:**
*   Generation of drafts of progress notes (SOAP) or discharge summaries, saving administrative time [3].
*   Conversion of (transcribed) voice notes into structured clinical documentation.

**Treatment Personalization:**
*   Creation of personalized exercise plans, adjusted for comorbidities (e.g., diabetes, osteoporosis) or environmental restrictions (e.g., exercises that can be done at home without equipment) [1].
*   Development of exercise progression and regression criteria based on the tissue healing phase.

**Education and Communication:**
*   Preparation of clear and concise answers to frequently asked patient questions (FAQs) about their condition or treatment [1].
*   Creation of content for social media or clinical presentations on physical therapy topics.

## Pitfalls
**Clinical Hallucinations:** AI can generate information that seems plausible but is clinically incorrect or outdated (hallucinations). **Always verify the AI's recommendations** with clinical reasoning and evidence-based guidelines [1] [2].

**Lack of Specificity:** Vague prompts (e.g., "What exercises for the shoulder?") result in generic responses of low clinical usefulness. The lack of details about the patient (age, comorbidities, injury phase) is a common mistake [1].

**Confidentiality Violation (HIPAA/LGPD):** Including patient identifying data (name, date of birth, address) in AI prompts is a serious violation of privacy and professional ethics. Anonymization is mandatory [1].

**Over-reliance:** Using AI as a substitute for clinical judgment. AI is a support tool, not a substitute for the physical therapist's assessment and decision-making [2].

**Ignoring the Output Format:** Failing to specify the format (table, list, SOAP note) can lead to disorganized responses that are difficult to integrate into the clinical workflow [3].

## URL
[https://www.physio-pedia.com/Physiopedia_AI_Assistant_Prompt_Writing_Guide](https://www.physio-pedia.com/Physiopedia_AI_Assistant_Prompt_Writing_Guide)
