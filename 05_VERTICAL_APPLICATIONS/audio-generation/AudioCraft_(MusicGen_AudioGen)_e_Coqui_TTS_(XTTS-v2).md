# AudioCraft (MusicGen/AudioGen) e Coqui TTS (XTTS-v2)

## Description

AudioCraft é uma biblioteca de código aberto do Meta AI para geração de áudio com aprendizado profundo, incluindo modelos como MusicGen (música) e AudioGen (efeitos sonoros). Coqui TTS é um kit de ferramentas de código aberto para Text-to-Speech (TTS), com o modelo XTTS-v2 que se destaca pela clonagem de voz e suporte multilíngue. Ambos representam soluções de ponta para a criação de conteúdo de áudio generativo.

## Statistics

AudioCraft: Desenvolvido pelo Meta AI, MusicGen treinado em 20.000 horas de música licenciada. Coqui TTS: Suporta mais de 1100 idiomas (modelos pré-treinados), XTTS-v2 suporta 17 idiomas, clonagem de voz com 6 segundos de áudio, inferência em streaming com < 200ms de latência.

## Features

AudioCraft: Geração de música a partir de texto e melodia, geração de efeitos sonoros, compressão de áudio de alta qualidade (EnCodec). Coqui TTS: Síntese de fala multilíngue, clonagem de voz com amostras curtas, transferência de emoção e estilo, suporte a diversos modelos de TTS.

## Use Cases

AudioCraft: Composição musical, trilhas sonoras para jogos/vídeos, prototipagem musical. Coqui TTS: Criação de audiolivros/podcasts, voz para assistentes virtuais/chatbots, localização de conteúdo, personalização de voz.

## Integration

A integração de ambos os projetos é feita através de bibliotecas Python (`audiocraft` e `TTS`). Exemplos de código demonstram a inicialização dos modelos e a geração de áudio a partir de texto, com o Coqui TTS exigindo um arquivo de áudio de referência para a clonagem de voz.

## URL

AudioCraft: https://github.com/facebookresearch/audiocraft; Coqui TTS: https://github.com/coqui-ai/TTS