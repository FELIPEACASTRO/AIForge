# Advanced Transfer Learning: PEFT Strategies (LoRA, Adapters)

## Description

Advanced Transfer Learning, focused on **Parameter-Efficient Fine-Tuning (PEFT)** techniques, is a crucial methodology for adapting Large Language Models (LLMs) and other pre-trained models to specific tasks. Instead of performing full fine-tuning of all the billions of parameters in the base model, PEFT introduces a small number of trainable parameters (the 'adapters') while keeping the vast majority of the original weights **frozen**. This allows the model to retain its general knowledge while learning specific nuances of the new task, resulting in fast, efficient, and high-performance adaptation.

## Statistics

PEFT techniques offer substantial efficiency gains. For example, **LoRA (Low-Rank Adaptation)** can reduce the number of trainable parameters by more than **99%** and GPU memory usage by up to **80%** compared to full fine-tuning. This enables training models with tens of billions of parameters on consumer hardware (16GB or 24GB GPUs), democratizing access to LLM fine-tuning.

## Features

The main PEFT techniques include:\n\n*   **Adapter Layers:** Injection of small *bottleneck* layers into the model architecture. Only the weights of these layers are trained.\n*   **LoRA (Low-Rank Adaptation):** Decomposition of weight matrices into two low-rank matrices. Only the weights of these low-rank matrices are optimized.\n*   **Prefix-Tuning:** Optimization of a small set of continuous vectors ('prefixes') that are concatenated to the input sequence, guiding the model toward the task.\n*   **Weight Freezing:** The core principle is to keep the pre-trained model's weights (the 'knowledge base') fixed, preserving stability and general knowledge.

## Use Cases

Advanced Transfer Learning is applied in:\n\n*   **LLM Adaptation:** Customizing models such as Llama, Mistral, or T5 for specific domain tasks (e.g., legal, medical, financial) with limited resources.\n*   **Multi-Task Learning:** Training a single base model for multiple tasks, where each task has its own set of PEFT adapters, allowing fast *switching* between tasks.\n*   **Deployment on Edge Devices:** Creating smaller, more efficient models for deployment on devices with memory and compute constraints.\n*   **Rapid Research:** Accelerated prototyping and experimentation with new tasks and datasets.

## Integration

Integration is largely facilitated by Hugging Face's **PEFT (Parameter-Efficient Fine-Tuning)** library, which abstracts the complexity of the different techniques. The typical workflow involves:\n\n1.  Installation: `pip install peft transformers`\n2.  Loading the Base Model:\n    ```python\n    from transformers import AutoModelForCausalLM\n    model = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\n    ```\n3.  Adapter Configuration (LoRA Example):\n    ```python\n    from peft import LoraConfig, get_peft_model\n    config = LoraConfig(\n        r=8, \n        lora_alpha=16, \n        target_modules=[\"q_proj\", \"v_proj\"], \n        lora_dropout=0.05,\n        bias=\"none\", \n        task_type=\"CAUSAL_LM\"\n    )\n    peft_model = get_peft_model(model, config)\n    peft_model.print_trainable_parameters()\n    # Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06220\n    ```\n4.  Training: The `peft_model` is trained like a standard `transformers` model, but only the LoRA parameters are updated.

## URL

https://huggingface.co/docs/peft/en/index