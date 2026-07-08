# Tree-of-Thoughts (ToT) Prompting

## Description

**Tree-of-Thoughts (ToT) Prompting** is an advanced prompt engineering technique that generalizes the Chain-of-Thought (CoT) method, enabling Large Language Models (LLMs) to perform a more deliberate and strategic decision-making process. Instead of following a single linear reasoning path (as in CoT), ToT structures the problem-solving process as a **search tree**, where each node represents a "thought" (a coherent unit of text, such as an intermediate step). This allows the LLM to explore multiple reasoning paths in parallel, evaluate the promise of each path, and make global choices, including the ability to look ahead (*lookahead*) or backtrack (*backtracking*) when necessary. ToT is particularly effective in tasks that require non-trivial planning, search, and complex reasoning, where the initial decisions are crucial to the final success [1] [2].

## Statistics

- **Performance Increase:** ToT demonstrated a significant increase in problem-solving capability compared to Chain-of-Thought (CoT).
- **Game of 24:** In the "Game of 24" challenge, GPT-4 with CoT solved only **4%** of the tasks, while GPT-4 with ToT achieved a success rate of **74%** [1].
- **Other Gains:** ToT also showed substantial improvements in tasks such as **Creative Writing** and **Mini Crosswords**, which require non-trivial planning and search [1].
- **Citation:** The original paper "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023) is a fundamental reference in the field, with a high citation rate since its publication [1].

## Features

- **Exploration of Multiple Paths:** Allows the LLM to generate and evaluate several reasoning sequences (thoughts) in parallel, instead of being limited to a single linear chain.
- **Strategic Search:** Uses search algorithms (such as **Depth-First Search (DFS)**, **Breadth-First Search (BFS)** and **Beam Search**) to navigate the tree of thoughts, selecting the most promising paths.
- **Self-Evaluation:** The LLM evaluates the quality and progression of each "thought" toward the final solution, using language-based reasoning to guide the search.
- **Deliberate Decision:** Facilitates strategic decision-making, allowing the model to look ahead and backtrack, overcoming the token-by-token decision limitations of CoT [1] [2].

## Use Cases

- **Puzzle and Game Solving:** Especially effective in games that require planning and search, such as the **Game of 24** and **Sudoku** (in framework variations) [1] [2].
- **Creative Writing:** Generation of narratives or texts that require coherence and long-term planning.
- **Multi-Step Reasoning:** Tasks that involve multiple interrelated variables and where the decision at one step critically affects subsequent steps.
- **Strategic Planning:** Simulation of decision-making processes that require looking ahead and evaluating different scenarios (e.g.: business planning, market analysis) [2].

## Integration

ToT can be implemented in two main ways: via code (integrating the LLM with search algorithms) or via prompt (instructing the LLM to simulate the search process).

**Simple Prompt Example (ToT Simulation):**
A simplified approach, proposed by Dave Hubert, instructs the LLM to simulate a group deliberation process to solve complex problems [2]:
```
Imagine three different experts are answering this question.
All experts will write down 1 step of their thinking,
and then share it with the group.
Then all experts will proceed to the next step, and so on.
If any expert realizes they are wrong at any point, they should leave.
The question is... [Insert the complex question here]
```

**Best Practices:**
- **Use for Complex Problems:** Apply ToT to tasks that require planning, strategy, and where CoT fails (e.g.: puzzles, multi-step reasoning).
- **Prompt Structuring:** For code-based implementations, the prompt should be structured to generate **coherent thoughts** (not just tokens), **evaluate the state** (heuristic) and **select the next step** [1].
- **Search Algorithms:** The choice of algorithm (DFS, BFS, Beam Search) should be adapted to the nature of the problem. DFS is useful for deeply exploring a path, while BFS/Beam Search are better for maintaining a diversity of options [2].

## URL

https://arxiv.org/abs/2305.10601