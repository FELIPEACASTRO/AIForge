# Prompt Tuning

## Description

**Prompt Tuning** is a *Parameter-Efficient Fine-Tuning* (PEFT) technique that adapts pre-trained Large Language Models (LLMs) while keeping their parameters frozen. The adaptation is done by optimizing a small set of trainable continuous vectors, called **soft prompts**, which are appended to the model's input. Unlike full *fine-tuning*, which adjusts millions or billions of parameters, Prompt Tuning adjusts only a few thousand parameters, drastically reducing computational costs and training time. The process involves initializing the *soft prompts* (usually with random values or word *embeddings*), running a training loop (*forward* and *backward pass*), and optimizing these vectors with a loss function until they reach the desired performance for the specific task. This approach enables the creation of modular, task-specific prompts, facilitating efficient deployment and adaptation to new domains.

## Statistics

- **Proven Efficiency:** The original Prompt Tuning (Lester et al., 2021) demonstrated performance comparable to *full fine-tuning* on T5-XXL models (11B parameters) on SuperGLUE tasks, tuning only 0.01% of the parameters.
- **P-Tuning v2:** A variation that achieves performance comparable to *full fine-tuning* across different model scales (330M to 10B parameters), demonstrating strong parameter efficiency.
- **Cost:** Drastic reduction in GPU consumption and training time compared to *full fine-tuning*.
- **Primary Citation:** Lester, B., Al-Rfou, R., & Constant, N. (2021). *The Power of Scale for Parameter-Efficient Prompt Tuning*. arXiv:2104.08691.

## Features

- **Parameter Efficiency:** Updates only a small subset of continuous vectors (soft prompts), keeping the base LLM frozen.
- **Modular Adaptation:** Prompts are task-specific, allowing the base model to be reused for multiple tasks with different *soft prompts*.
- **Soft Prompts:** Continuous vectors in the *embedding* space, unconstrained by vocabulary, which allows more flexible and effective optimization than *hard prompts* (natural text).
- **Cost Reduction:** Significantly decreases the memory and computation requirements for training and deployment.
- **Framework Flexibility:** Supports knowledge transfer and composition mechanisms, serving as the basis for variations such as P-Tuning, Prefix Tuning, and Decomposed Prompt Tuning.

## Use Cases

- **Domain Adaptation:** Tuning a generic LLM for specific domains (e.g., medicine, law, finance) with a limited set of domain-specific data.
- **Specific Content Generation:** Training the model to generate specific types of text, such as product descriptions for e-commerce, news summaries, or answers to questions in a standardized format.
- **Text Classification:** Tasks such as sentiment analysis, document classification, and content moderation.
- **Low-Resource Scenarios (*Few-Shot Learning*):** Ideal for situations where little labeled data is available, since optimizing the *soft prompts* is more efficient at extracting knowledge from the pre-trained model.
- **On-Device Deployment:** Its memory and compute efficiency makes it suitable for deployment in resource-constrained environments.

## Integration

Prompt Tuning is implemented through PEFT libraries, such as Hugging Face's. Best practices include:

**1. Implementation Example (Python/PEFT):**
The Prompt Tuning configuration is done by defining the number of virtual tokens and the task type:
```python
from peft import PromptTuningConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompt Tuning configuration
config = PromptTuningConfig(
    num_virtual_tokens=50,  # 20-50 for classification, 50-100+ for generation
    task_type="CAUSAL_LM",  # E.g.: Text Generation
    prompt_tuning_init="RANDOM" # Initialization
)

# Load model and add Prompt Tuning capability
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = get_peft_model(model, config)
```

**2. Best Practices:**
- **Prompt Size:** The number of virtual tokens (*num_virtual_tokens*) is a crucial hyperparameter. It should be tuned: 20-50 for classification tasks and 50-100 or more for complex text generation tasks.
- **Verbalizers:** For classification tasks, the use of *verbalizers* (mapping the model's output to class labels) is essential.
- **Hyperparameter Sensitivity:** Prompt Tuning is sensitive to the learning rate and the initialization method.
- **Model Scale:** The effectiveness of Prompt Tuning is greatest in large-scale models (above 10B parameters), where it can match the performance of *full fine-tuning*.

## URL

https://arxiv.org/html/2507.06085v2
