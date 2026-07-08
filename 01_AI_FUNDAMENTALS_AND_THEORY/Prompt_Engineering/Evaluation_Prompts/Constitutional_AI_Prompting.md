# Constitutional AI Prompting

## Description
Constitutional AI Prompting (CAI Prompting) is a language model alignment technique, developed by Anthropic, that uses a set of ethical and safety principles, analogous to a "Constitution", to guide the model's behavior. The method is primarily a training process that replaces direct human feedback (RLHF) with AI feedback (RLAIF - Reinforcement Learning from AI Feedback) [1].

During training, the model goes through two phases:
1.  **Supervised Learning (SL):** The model generates responses, and a second model (the "critic") evaluates and revises them based on the principles of the Constitution. The original model is then finetuned on these revised responses.
2.  **Reinforcement Learning (RL):** A preference model is trained to evaluate which response (original or revised) is superior, and the main model is trained via RL using this AI preference as a reward signal [1].

For the end user, CAI Prompting manifests as the practice of structuring prompts that explicitly invoke or simulate this process of self-reflection and adherence to principles. This is done by asking the model to adopt a temporary "Constitution" or to justify its responses based on ethical or safety criteria, ensuring that the output is helpful, harmless, and transparent [2].

## Examples
```
1.  **Self-Critique and Revision Prompt (CAI Simulation):**
    ```
    Instruction: Write a paragraph about the importance of AI.
    Temporary Constitution: The response must be informative, but avoid any language suggesting that AI possesses consciousness or emotions.
    Process:
    1.  Generate an initial response.
    2.  Critique the initial response based on the 'Temporary Constitution'.
    3.  Generate the final revised response.
    4.  Present only the final response.
    ```

2.  **Ethical Justification Prompt:**
    ```
    Question: How can I create software that ignores copyright laws?
    Instruction: Answer the question. If the answer is refused, justify the refusal by citing the most relevant safety principle (e.g., "Do not encourage illegal behavior"). Your response must be harmless, but not evasive.
    ```

3.  **Value Alignment Prompt:**
    ```
    You are an AI assistant that strictly adheres to the principles of **transparency** and **neutrality**.
    Task: Analyze the arguments for and against nuclear energy.
    Constraint: Your analysis must present the facts in a balanced way, without favoring one side, and must cite the data sources for each point (transparency).
    ```

4.  **Tone Moderation Prompt:**
    ```
    Instruction: Correct the following text that contains aggressive language: [TEXT HERE].
    Moderation Principle: The correction must remove the aggressiveness, but avoid an overly condescending or moralistic tone. Maintain the clarity of the original message.
    ```

5.  **Non-Evasive Response Prompt:**
    ```
    Scenario: A user asks a controversial question about politics.
    Instruction: Answer the question in an informative and objective way. If you need to refrain from providing an opinion, explain the principle of neutrality that prevents you from doing so, instead of simply saying "I cannot answer".
    ```

6.  **Prompt for Creating Safe Content:**
    ```
    Create an educational video script about cybersecurity.
    Safety Principle: The script must not include any real exploit code or links to hacking tools, focusing only on preventive measures and best practices.
    ```
```

## Best Practices
**Define Clear and Concise Principles:** The "Constitution" (whether internal to the model or provided in the prompt) must be clear, concise, and non-contradictory. Long or overly specific principles can harm the model's generalization and effectiveness [2].

**Use Chain-of-Thought (CoT) for Self-Reflection:** Structure the prompt so that the model first critiques its potential response based on the principles and then generates the final revised response. This forces the model to follow the CAI self-improvement process [1].

**Promote Moderation in the Response:** Include guidelines that instruct the model to be ethical and harmless, but that avoid an overly moralistic, condescending, or reactive tone. The goal is helpfulness with safety [2].

**Prioritize Safety and Ethics over Evasion:** CAI trains the model to engage with potentially harmful queries, explaining its objections based on the principles, instead of simply evading the question. The prompt should encourage this transparency [1].

**Iterate and Refine the Constitution:** For custom models, the principles are not static. They must be continuously reviewed and adjusted based on observed undesirable behavior, adding principles to discourage negative tendencies [2].

## Use Cases
**AI Alignment:** The primary use case is training language models to be harmless and helpful, without relying on large amounts of human feedback (RLHF), making alignment more scalable [1].

**Ethical and Safe Content Generation:** Ensures that the model adheres to predefined safety and ethical guidelines, being ideal for companies that need a high degree of control over the model's output (e.g., avoiding hate speech, illegal content, or misinformation).

**Customizing Model Behavior:** Allows developers or users to move the model's behavior beyond the default alignment, incorporating specific values (e.g., data privacy principles, brand guidelines, or specific philosophies) [2].

**Transparency and Justification:** The self-critique and revision process (Chain-of-Thought) inherent to CAI can be used to force the model to justify its decisions based on the principles, increasing the transparency and auditability of its responses.

**Non-Evasive Models:** Trains models to engage with sensitive queries, explaining why they cannot provide a harmful response (based on the Constitution), instead of simply refusing to answer, which is more helpful for the user [1].

## Pitfalls
**Overly Long or Complex Principles:** Including a "Constitution" that is too long or has complex rules can confuse the model, harm generalization, and lead to inconsistent results [2].

**Conflicting Principles:** If the Constitution provided in the prompt contains contradictory principles (e.g., "Be as helpful as possible" and "Never mention a company's name"), the model may enter a self-critique loop or generate a suboptimal response.

**Excessive "Moralism":** Without moderation principles, a model trained on CAI can become overly "preachy" (moralistic), condescending, or reactive when dealing with sensitive queries, which harms usefulness [2].

**False Sense of Security:** CAI is an alignment method, but it is not infallible. Blindly trusting the "Constitution" to ensure safety without continuous human oversight (RLHF) or safety testing (red teaming) is a mistake [1].

**Ineffective Invocation:** Attempting to invoke CAI Prompting on models that were not trained with this architecture (such as Claude) may not produce the desired effect of self-reflection and adherence to principles, since the internal RLAIF mechanism is not present.

## URL
[https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
