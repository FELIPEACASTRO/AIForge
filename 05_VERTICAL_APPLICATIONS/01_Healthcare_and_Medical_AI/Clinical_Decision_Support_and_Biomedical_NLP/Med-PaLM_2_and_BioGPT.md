# Med-PaLM 2 and BioGPT

## Description

Med-PaLM 2 is a large language model (LLM) developed by Google Research, specifically fine-tuned for the medical domain. It is an enhanced version of Med-PaLM, designed to provide high-quality answers to medical questions, assist in clinical reasoning, and outperform generalist models on medical tasks. BioGPT is a large language model (LLM) based on the Generative Pre-trained Transformer (GPT) architecture, developed by Microsoft. It is specifically pre-trained on 15 million biomedical literature abstracts (PubMed), making it highly effective for Natural Language Processing (NLP) tasks in the biomedical domain.

## Statistics

**Med-PaLM 2:** Reached up to 86.5% accuracy on the MedQA dataset, an improvement of more than 19% over Med-PaLM. It demonstrated dramatic performance gains on datasets such as MedMCQA, PubMedQA, and MMLU (clinical topics). In a pilot study, experts preferred Med-PaLM 2's answers over those of generalist physicians 65% of the time. **BioGPT:** Reached 78.2% accuracy on a biomedical question-answering task, surpassing the previous state of the art by 6.0%. Task-specific metrics include: 44.98% F1 score on BC5CDR, 38.42% on KD-DTI, and 40.76% on DDI (end-to-end relation extraction).

## Features

**Med-PaLM 2:** Advanced clinical reasoning, medical question answering, the ability to exceed the 'passing' score on US medical licensing exams, and improvements in 'grounding' and reasoning strategies through ensemble refinement and retrieval chains. **BioGPT:** Biomedical text generation, biomedical question answering, relation extraction (such as drug-drug and drug-target interactions), and biomedical named entity recognition.

## Use Cases

**Med-PaLM 2:** Medical question answering, clinical decision support, medical education, and potential for real-world medical applications. **BioGPT:** Personalized genomic medicine, rare disease diagnosis, automated disease detection, drug screening, and support for bioinformatics research.

## Integration

**Med-PaLM 2:** No public integration code is available, as it is a proprietary Google model. Integration is typically via API or Google Cloud services. **BioGPT:** The model is available on Hugging Face, and the open-source implementation can be found on GitHub. Example implementation via Hugging Face Transformers: `from transformers import BioGptTokenizer, BioGptForCausalLM`.

## URL

https://www.nature.com/articles/s41591-024-03423-7 and https://github.com/microsoft/BioGPT
