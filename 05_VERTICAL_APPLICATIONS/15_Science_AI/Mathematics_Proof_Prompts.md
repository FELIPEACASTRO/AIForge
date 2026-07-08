# Mathematics Proof Prompts

## Description
The **Mathematics Proof Prompts** technique refers to the engineering of specific instructions for large language models (LLMs) with the goal of generating, verifying, or refuting **formal and rigorous mathematical proofs**. Unlike prompts that seek only the final numerical answer, this technique focuses on the **chain of logical reasoning** and the **formal structure** of the demonstration. The goal is to mitigate the tendency of LLMs to make logical errors, unjustified assumptions, or over-generalize patterns observed in smaller cases [1]. The effectiveness of this technique lies in forcing the model to emulate a mathematician's reasoning process, requiring clarity, coherence, and the use of formal notation (such as LaTeX) to ensure the precision of the output. Recent studies, such as those evaluating LLMs on Mathematics Olympiad problems, demonstrate that well-structured prompts are crucial for achieving acceptable performance in high-level reasoning tasks [1].

## Examples
```
1. **Formal Proof Generation (Main Prompt from the Article):**
```
Give a thorough answer to the following question. Your answer will be graded by human judges based on accuracy, correctness, and your ability to prove the result. You should include all steps of the proof. Do not skip important steps, as this will reduce your grade. It does not suffice to merely state the result. Use LaTeX to format your answer.

**Problem:** Prove that, for every natural number $n$, the sum of the first $n$ odd numbers is equal to $n^2$.
```

2. **Proof Verification and Evaluation (Judge Prompt):**
```
# Instruction
You are an expert mathematician that grades solutions of high-school olympiad-level problems. You will be given a mathematical problem, as well as a grading scheme that you should adhere to. Your task is to accurately grade a solution according to that grading scheme.

# Problem and Scheme
## Problem: Prove that $\sqrt{2}$ is irrational.
## Grading Scheme:
- 7 points: Complete and rigorous proof by contradiction.
- 4 points: Proof with a minor logical flaw or calculation error.
- 2 points: Correct attempt at contradiction, but incomplete.
- 0 points: Incorrect answer or no attempt at proof.

# Solution to Grade
## Solution: [Insert the student's solution here]
```

3. **Proof by Induction (Explicit Step by Step):**
```
Use the **Proof by Induction** method to demonstrate the following proposition. Present your answer in three clear sections: Base Case, Induction Hypothesis, and Inductive Step. Use LaTeX format for all mathematical expressions.

**Proposition:** $\sum_{i=1}^{n} i^3 = \left(\frac{n(n+1)}{2}\right)^2$
```

4. **Proof by Contradiction (CoT Reasoning Instruction):**
```
You must prove the statement below using the **Proof by Contradiction** method. Before presenting the final proof, use the Chain-of-Thought (CoT) technique to detail your reasoning.

1. **Contradiction Assumption:** State the negation of the statement.
2. **Logical Development (CoT):** Show the sequence of logical steps that lead to a contradiction.
3. **Formal Conclusion:** State the final conclusion.

**Statement:** There is no largest prime number.
```

5. **Lemma Generation and Auxiliary Proof (Few-Shot):**
```
**Instruction:** Given the main statement, first suggest an auxiliary lemma that can simplify the proof. Then, prove the lemma and use it to prove the main statement.

**Main Statement:** If $n$ is an integer, then $n^2 + n$ is always even.

**Lemma Example (Few-Shot):**
*Lemma:* An integer $n$ is even if and only if $n=2k$ for some integer $k$. An integer $n$ is odd if and only if $n=2k+1$ for some integer $k$.

**Your Task:**
1. Suggest Auxiliary Lemma.
2. Proof of the Lemma.
3. Proof of the Main Statement using the Lemma.
```

6. **Proof Refutation (Logical Flaw Identification):**
```
Analyze the proof presented below for the statement "Every triangle is isosceles". If the proof is incorrect, identify the **first logical error** or the **invalid assumption** and explain why it invalidates the proof.

**Proof to be Refuted:** [Insert here the classic flawed proof, such as the one that uses the intersection of the angle bisector and the perpendicular bisector.]
```

7. **Translation to Formal Language (Lean/Isabelle):**
```
Translate the following informal proof into a sequence of steps that can be verified by a formal proof system (such as Lean or Isabelle). Focus on precision and logical syntax.

**Informal Proof:** "The composition of two injective functions is injective. Let $f: A \to B$ and $g: B \to C$ be injective. To prove that $g \circ f$ is injective, assume that $(g \circ f)(x_1) = (g \circ f)(x_2)$. This means $g(f(x_1)) = g(f(x_2))$. Since $g$ is injective, we have $f(x_1) = f(x_2)$. Since $f$ is injective, we have $x_1 = x_2$. Therefore, $g \circ f$ is injective."
```

8. **Proof Generation with Method Restriction:**
```
Generate a proof for the following statement, but **strictly prohibit** the use of the Fundamental Theorem of Calculus. The proof must be based solely on Riemann sums and limits.

**Statement:** Compute the definite integral $\int_{0}^{1} x^2 dx$.
```
```

## Best Practices
**1. Require a Complete and Rigorous Proof:** Explicitly include in the prompt the need to present **all steps of the proof** and the prohibition against skipping steps, ensuring logical rigor [1].
**2. Specify the Output Format:** Require the use of mathematical formatting languages, such as **LaTeX**, to ensure the clarity and precision of mathematical expressions and symbols [1].
**3. Use an Evaluation Context (Judge Prompt):** Add a system context (or in the prompt) that simulates an evaluation by a "human judge" or "expert mathematician". This raises the LLM's reasoning standard, encouraging it to be more cautious and rigorous [1].
**4. Apply Structured Chain-of-Thought (CoT):** For complex problems, instruct the model to detail its reasoning in logical steps (e.g., "Contradiction Assumption", "Logical Development", "Formal Conclusion") before presenting the final proof.
**5. Auxiliary Proof (Few-Shot):** Provide examples of proofs or auxiliary lemmas (Few-Shot) to guide the model in the desired proof strategy (e.g., proof by induction, contradiction) [1].
**6. Method Restriction:** To test the model's flexibility and fundamental knowledge, explicitly prohibit the use of advanced theorems or methods, forcing it to build the proof from basic principles.

## Use Cases
**1. Formal Proof Generation:** Creating rigorous and detailed solutions for advanced-level mathematics problems (e.g., Mathematics Olympiads, Fundamental Theorems) [1].
**2. Automated Solution Evaluation:** Using LLMs as "judges" (Judge Prompts) to evaluate the correctness, rigor, and clarity of proofs submitted by students or other models [1].
**3. Translation to Formal Proof Systems:** Converting informal proofs in natural language into formal languages (e.g., Lean, Isabelle), facilitating computer verification.
**4. Research in Mathematical Reasoning:** Studying the reasoning capabilities and failures of LLMs, identifying areas where the model fails (logic, assumption, creativity) for future improvements [1].
**5. Tutoring and Education:** Generating step-by-step proofs for instructional purposes, helping students understand the structure and logic behind mathematical demonstrations.

## Pitfalls
**1. Excessive Pattern Generalization:** The tendency of LLMs to over-generalize patterns observed in smaller numerical cases to larger cases, without providing a formal proof that supports the statement [1].
**2. Non-Existent Citation:** Fabrication of references, theorems, or lemmas that seem plausible but are false or unverifiable, in order to lend credibility to the proof [1].
**3. Hidden Logical Flaws:** The proof may appear superficially correct but contain logical fallacies or unjustified assumptions that invalidate the conclusion.
**4. Lack of Structural Clarity:** Variation in the coherence and structure of the solution, making step-by-step verification of the reasoning difficult.
**5. Answer Boxing:** The model may focus on providing the final answer in a "boxed" format (boxed answer), a training artifact, instead of concentrating on the rigor of the proof [1].

## URL
[https://arxiv.org/pdf/2503.21934](https://arxiv.org/pdf/2503.21934)
