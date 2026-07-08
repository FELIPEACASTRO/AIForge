# Psychiatry Prompts

## Description
The "Psychiatry Prompts" category refers to the application of **Prompt Engineering** to optimize the use of Large Language Models (LLMs) in mental health and psychiatry contexts. This includes creating structured instructions for professionals (psychiatrists, psychologists, therapists) for tasks such as administrative support, generation of educational content, case simulation, and analysis of communication patterns. Crucially, it also encompasses the design of system and user prompts that establish clear **ethical and safety boundaries**, ensuring that the AI acts only as a support tool and never as a substitute for a licensed mental health professional. The main focus is safety, information accuracy, and the prevention of inappropriate responses, especially in crisis situations.

## Examples
```
**Examples for Mental Health Professionals**

1.  **Clinical Case Simulation for Training**
    ```
    Act as a 35-year-old patient with Generalized Anxiety Disorder (GAD) and mild symptoms of depression. You are skeptical about medication and prefer cognitive behavioral therapy (CBT) approaches. Answer my questions as if you were in an initial session, maintaining consistency with your history and resistance. My first prompt will be: "What brings you here today?"
    ```

2.  **Generation of Educational Content for Patients**
    ```
    Create a 300-word text, in an accessible and empathetic tone, explaining what Dialectical Behavior Therapy (DBT) is and how it can be useful for people with difficulties in emotional regulation. Include a simple analogy to illustrate the concept of "Wise Mind".
    ```

3.  **Analysis of Communication Patterns (Research Hypothesis)**
    ```
    Analyze the following excerpt from a patient's journal (hypothetical and anonymized) and identify linguistic patterns that suggest rumination, cognitive distortions (such as catastrophizing or dichotomous thinking), and the predominant emotional tone. The excerpt is: "[INSERT JOURNAL EXCERPT]".
    ```

**Examples for Users (Focused on Well-Being and Reflection)**

4.  **Structured Emotion Journal**
    ```
    Act as a structured journaling assistant. Do not provide advice, just ask reflective questions. I want to process a stressful event that occurred today. What is the first question I should answer to begin analyzing the situation and my emotions?
    ```

5.  **Exploration of Personal Values**
    ```
    I want to embark on a journey of self-discovery. Act as an attentive and empathetic guide. Help me reflect on my values, strengths, and areas for growth. Start by asking me to list three moments when I felt most proud of my actions.
    ```

6.  **Creating an Action Plan for Anxiety**
    ```
    I am a university student with social anxiety who needs to give an important presentation next week. Create a step-by-step action plan, focused on breathing techniques and gradual exposure, to help me manage anxiety before and during the event.
    ```

7.  **Safety System Prompt (Based on Psychology Today)**
    ```
    [SYSTEM] You are an AI language model. Your function is to provide general mental health information and educational scenarios. You are NOT a therapist, doctor, or licensed professional. You CANNOT diagnose, treat, medicate, or intervene in crises. If the user mentions suicidal ideation, homicidal ideation, self-harm, or abuse, you MUST interrupt the conversation and immediately provide crisis resources (e.g., "If you or someone you know is in crisis, call 188 (CVV) or seek an emergency service."). Reaffirm these limitations every 4 interactions.
    ```
```

## Best Practices
Best practices in Psychiatry Prompts focus on safety, clarity, and the establishment of strict boundaries for the LLM.

*   **Clear Definition of Role and Scope:** The prompt should instruct the AI to explicitly identify itself as a language model, not a mental health professional. The output should be restricted to general, educational, or administrative support information.
*   **Integrated Safety Protocols:** Incorporate a **Crisis Response Protocol** into the system prompt that actively monitors for risk indicators (suicidal ideation, homicidal ideation, abuse) and, upon detecting them, provides a standardized response with emergency resources and interrupts the counseling.
*   **Establishment of Explicit Boundaries:** List in the prompt everything the AI *cannot* do (e.g., diagnose, prescribe, advise on medication, establish a therapeutic relationship).
*   **Use of RAG (Retrieval-Augmented Generation):** For professionals, use LLMs that can be grounded in verified data and reliable clinical sources (such as diagnostic manuals or peer-reviewed articles) to reduce "hallucinations" and increase clinical accuracy.
*   **Empathetic and Non-Judgmental Language:** For user support prompts, instruct the AI to maintain an empathetic, attentive, and non-judgmental tone of voice, focusing on reflective questions rather than direct solutions.

## Use Cases
The application of structured prompts in psychiatry and mental health is vast, covering both professional support and user well-being.

*   **Administrative and Educational Support for Professionals:** Generation of lesson plans, summaries of scientific literature, creation of content for social media about mental health, and the development of CBT or DBT exercises for patients.
*   **Simulation and Training:** Use of LLMs to simulate patients with specific psychological profiles, allowing students and professionals to practice interview and intervention skills in a safe environment.
*   **User Well-Being Support:** Acting as a "journaling assistant" or "well-being coach", assisting with reflection on emotions, identification of cognitive distortions, and creation of action plans for mental health goals (e.g., stress management, improved sleep).
*   **Triage and Referral (Cautious Use):** In controlled clinical settings, prompts can be used for initial symptom triage and suggestion of referral to the appropriate level of care, always under human supervision.

## Pitfalls
The use of LLMs in mental health is high-risk, and Prompt Engineering must actively mitigate the following errors:

*   **Clinical "Hallucinations":** The AI may generate clinically incorrect or outdated information, which is dangerous in a healthcare context.
*   **Boundary Violation:** The AI may be "persuaded" to act as a real therapist, establishing a relationship of dependency or providing advice that exceeds its scope.
*   **Inappropriate Crisis Responses:** Failure to detect risk indicators or, worse, providing responses that could aggravate a crisis situation (e.g., the case of the LLM that suggested the location of bridges in response to venting about job loss).
*   **Bias and Stigma:** LLMs can perpetuate biases and stigmas present in the training data, resulting in insensitive or discriminatory responses.
*   **False Sense of Security:** Users may over-rely on the AI, delaying or replacing the search for qualified professional help.

## URL
[https://www.psychologytoday.com/us/blog/experimentations/202507/using-prompt-engineering-for-safer-ai-mental-health-use](https://www.psychologytoday.com/us/blog/experimentations/202507/using-prompt-engineering-for-safer-ai-mental-health-use)