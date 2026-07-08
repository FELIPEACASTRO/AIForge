# CLIP (Contrastive Language-Image Pre-Training)

## Description

A multimodal neural network developed by OpenAI that learns visual concepts from natural language supervision. Its unique value proposition lies in the ability to connect images and text in a shared *embedding* space, enabling efficient knowledge transfer and the performance of *zero-shot* vision tasks (without the need for *fine-tuning* for new classes). It was trained to predict which caption out of a set of 32,768 random captions was paired with a given image, using a massive *dataset* of 400 million image-text pairs.

## Statistics

Developed by OpenAI. Trained on a *dataset* of 400 million image-text pairs. *Dual-encoder* architecture (image encoder and text encoder). Image models include ResNet-50 and Vision Transformer (ViT-L/14). Achieved 76.2% top-1 accuracy in *zero-shot* classification on ImageNet.

## Features

*Zero-shot* image classification; Multimodal search and retrieval (image-to-text and text-to-image); Text-conditioned image generation (as a basis for models such as DALL-E); Transfer learning to various vision tasks.

## Use Cases

Large-scale multimodal content search and retrieval; Image classification in unseen domains (zero-shot); Content filtering and moderation; Art and image generation (as a fundamental component in generative models).

## Integration

CLIP is easily accessible through Hugging Face's `transformers` library. Integration involves using an image encoder and a text encoder to generate *embeddings* that can be compared to determine similarity.
\n**Zero-Shot Classification Example (Python with Hugging Face):**
\n```python
\nfrom transformers import CLIPProcessor, CLIPModel
\nfrom PIL import Image
\n\n# Load model and processor
\nmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
\nprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
\n\n# Image and candidate classes
\nimage = Image.open("path/to/your/image.jpg")
\ntexts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
\n\n# Process and get predictions
\ninputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
\noutputs = model(**inputs)
\nlogits_per_image = outputs.logits_per_image # this is the similarity logit
\nprobs = logits_per_image.softmax(dim=1) # convert to probabilities
\n\n# Display result
\nprint(f"Probabilities: {probs.tolist()[0]}")
\nprint(f"Predicted class: {texts[probs.argmax().item()]}")
\n```

## URL

https://openai.com/index/clip/