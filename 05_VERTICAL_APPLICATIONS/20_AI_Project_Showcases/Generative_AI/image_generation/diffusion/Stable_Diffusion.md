# Stable Diffusion

## Description

O Stable Diffusion é um modelo de difusão latente de texto para imagem de código aberto, desenvolvido pela Stability AI e colaboradores. Ele é notável por sua capacidade de gerar imagens fotorrealistas de alta resolução a partir de prompts de texto, mantendo a eficiência computacional ao operar no espaço latente, em vez do espaço de pixel. Sua natureza de código aberto fomentou um ecossistema massivo de ferramentas, interfaces de usuário e modelos ajustados.

## Statistics

**Arquitetura:** Modelo de Difusão Latente (LDM). **Parâmetros:** Versões recentes como SDXL possuem 3.5 bilhões de parâmetros. **Resolução:** Suporte nativo a resoluções de 1024x1024 pixels (SDXL) e 1 megapixel (SD3.5 Large). **Ecossistema:** Um dos maiores ecossistemas de código aberto em IA generativa, com milhares de modelos ajustados e interfaces de usuário (UIs) como Automatic1111.

## Features

Geração de texto para imagem (text-to-image), Imagem para imagem (image-to-image), Inpainting e Outpainting, Ajuste fino (Fine-tuning) com dados personalizados (ex: LoRA, DreamBooth), Suporte a múltiplos estilos (3D, fotografia, pintura, arte linear), Alta aderência ao prompt (especialmente nas versões 3.5).

## Use Cases

**Design Gráfico e Arte Digital:** Criação rápida de ativos visuais, conceitos de arte e ilustrações. **Mídia e Entretenimento:** Geração de fundos, texturas e personagens para jogos e filmes. **E-commerce e Publicidade:** Criação de imagens de produtos em diferentes cenários (fotografia de produto virtual). **Pesquisa:** Ferramenta para estudar modelos de difusão e desenvolver novas arquiteturas de IA.

## Integration

A integração é tipicamente feita através da biblioteca **Hugging Face Diffusers** em Python ou através da **Stability AI API**.

**Exemplo de Integração (Python com Diffusers):**
```python
import torch
from diffusers import DiffusionPipeline

# Carregar o pipeline do SDXL-Turbo (otimizado para velocidade)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
)
pipe.to("cuda") # Mover para GPU

prompt = "A high-quality, photorealistic image of a majestic lion wearing a crown, digital art."

# Gerar a imagem
image = pipe(
    prompt=prompt, 
    num_inference_steps=1, 
    guidance_scale=0.0
).images[0]

image.save("generated_image_sdxl_turbo.png")
```

## URL

https://stability.ai/stable-image