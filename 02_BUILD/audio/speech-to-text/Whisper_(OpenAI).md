# Whisper (OpenAI)

## Description

O Whisper é um sistema de Reconhecimento Automático de Fala (ASR) de propósito geral e código aberto, desenvolvido pela OpenAI. Sua principal proposta de valor reside na sua **robustez e universalidade**, sendo treinado em um vasto conjunto de dados de 680.000 horas de áudio supervisionado multilingue e multitarefa. Essa diversidade de treinamento permite que o modelo lide com ruídos de fundo, sotaques e diferentes formatos de áudio com alta precisão, superando muitos modelos ASR anteriores. Além da transcrição de fala para texto, o Whisper é capaz de realizar a tradução de fala de idiomas suportados para o inglês.

## Statistics

O modelo foi treinado em **680.000 horas** de dados de áudio diversos. Suporta a transcrição e tradução em **mais de 90 idiomas**. A OpenAI disponibiliza o modelo em várias versões (tiny, base, small, medium, large), permitindo um *trade-off* entre velocidade de inferência e precisão, com o modelo `large` sendo o mais preciso.

## Features

As principais capacidades do Whisper incluem **transcrição de alta precisão** (ASR), **tradução de fala** para o inglês, e **multitarefa** (combinando ASR e tradução). Sua arquitetura baseada em Transformer garante robustez contra variações de áudio e a capacidade de processar longas sequências de fala de forma coerente.

## Use Cases

O Whisper é amplamente utilizado para **legendagem automática** de vídeos e transmissões ao vivo, **transcrição de reuniões** e entrevistas para documentação, **criação de dados de treinamento** para outros modelos de Processamento de Linguagem Natural (PNL) e melhoria da **acessibilidade global** através da tradução de conteúdo de áudio.

## Integration

A integração pode ser feita através da **biblioteca Python de código aberto `openai-whisper`** ou da **API oficial da OpenAI**. A biblioteca de código aberto permite a execução local do modelo em hardware compatível. \n\n**Exemplo de Integração (Biblioteca Python):**\n```python\nimport whisper\n\n# Carrega o modelo (ex: 'base')\nmodel = whisper.load_model(\"base\")\n\n# Transcreve um arquivo de áudio\nresult = model.transcribe(\"audio.mp3\")\n\n# Imprime o texto transcrito\nprint(result[\"text\"])\n```

## URL

https://github.com/openai/whisper