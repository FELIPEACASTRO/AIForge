# LLaVA (Large Language and Vision Assistant)

## Description

O **LLaVA (Large Language and Vision Assistant)** é um modelo multimodal de código aberto que combina um codificador de visão (geralmente CLIP ou um modelo baseado em ViT) com um Large Language Model (LLM) de código aberto (como Vicuna ou Llama) para permitir a compreensão e o raciocínio sobre imagens e texto. Sua proposta de valor única reside em ser uma alternativa de código aberto de alto desempenho ao GPT-4V(ision), alcançando capacidades de conversação visual de nível comparável por meio de uma técnica chamada *Visual Instruction Tuning*. O LLaVA é treinado em um conjunto de dados multimodal gerado por GPT-4, o que lhe permite seguir instruções complexas que envolvem tanto a análise visual quanto a geração de linguagem natural. As versões mais recentes, como LLaVA-1.5 e LLaVA-NeXT, continuam a aprimorar a eficiência e o desempenho, tornando-o uma ferramenta fundamental para pesquisa e desenvolvimento de MLLMs (Multimodal Large Language Models) de código aberto.

## Statistics

**Parâmetros:** Disponível em tamanhos de 7 Bilhões (7B) e 13 Bilhões (13B) de parâmetros. **Desempenho:** A versão LLaVA-1.5 alcançou um desempenho de ponta em 11 benchmarks de visão-linguagem, superando modelos anteriores de código aberto e aproximando-se do desempenho do GPT-4V em várias tarefas. **Conjunto de Dados:** Treinado em um conjunto de dados de 150K pares de instruções visuais e de linguagem gerados por GPT-4. **Velocidade:** Otimizado para inferência, com implementações como vLLM que oferecem alta taxa de transferência (throughput).

## Features

**Arquitetura Modular:** Combina um codificador de visão (e.g., CLIP ViT-L/14) e um LLM (e.g., Vicuna, Llama 2) conectados por uma camada de projeção linear. **Visual Instruction Tuning:** Treinado em um conjunto de dados de instruções visuais de alta qualidade gerado por GPT-4 para alinhar as capacidades de visão e linguagem. **Capacidade de Alto Desempenho:** Alcança resultados de ponta em 11 benchmarks de visão-linguagem, incluindo VQA, GQA e ScienceQA, com modelos de 7B e 13B de parâmetros. **Suporte a Múltiplas Resoluções:** Versões recentes (LLaVA-NeXT) suportam diferentes proporções de aspecto e resoluções de imagem, melhorando a capacidade de processar imagens de alta resolução. **Inferência Otimizada:** Compatível com frameworks de inferência rápida como vLLM e Ollama, permitindo implantação e escalabilidade eficientes.

## Use Cases

**Assistente de Conversação Visual:** Responder a perguntas complexas sobre o conteúdo de uma imagem, como "O que está acontecendo nesta cena e por que o objeto X está lá?". **Análise de Documentos e Imagens:** Extração de informações de gráficos, tabelas e documentos digitalizados, permitindo a automação de processos de negócios. **Robótica e Visão Computacional:** Fornecer raciocínio de alto nível para sistemas robóticos, ajudando-os a entender o ambiente visual e a planejar ações. **Geração de Legendas (Captioning) e Descrições Detalhadas:** Criar descrições ricas e contextuais para imagens, útil para acessibilidade e catalogação de conteúdo. **Educação e Tutoria:** Explicar conceitos visuais em livros didáticos ou diagramas com base em perguntas do usuário.

## Integration

A integração do LLaVA é facilitada por sua natureza de código aberto e compatibilidade com ecossistemas de ML populares. **1. Hugging Face Transformers:** O método mais comum é usar a biblioteca `transformers` para carregar e executar o modelo.
```python
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image

model_id = "llava-hf/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id)

# Carregar imagem e prompt
image = Image.open("caminho/para/sua/imagem.jpg")
prompt = "USER: Descreva esta imagem em detalhes. ASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Gerar resposta
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```
**2. vLLM (Serviço de API Compatível com OpenAI):** O vLLM pode ser usado para servir o LLaVA com alta taxa de transferência, expondo uma API compatível com a do OpenAI, o que permite a integração com ferramentas como LangChain e clientes OpenAI.
```bash
# Servir o modelo com vLLM
python3 -m vllm.entrypoints.openai.api_server --model llava-hf/llava-1.5-7b-hf
```
**3. Ollama:** Para uso local e fácil, o Ollama oferece uma maneira simples de baixar e executar o LLaVA.
```bash
# Baixar e executar com Ollama
ollama run llava
```

## URL

https://llava-vl.github.io/