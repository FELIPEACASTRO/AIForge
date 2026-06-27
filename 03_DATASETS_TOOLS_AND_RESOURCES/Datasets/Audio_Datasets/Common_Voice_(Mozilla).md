# Common Voice (Mozilla)

## Description
O **Common Voice** é um projeto da Mozilla Foundation que visa construir o maior e mais diversificado corpus de voz aberto do mundo. O dataset é composto por clipes de voz gravados por voluntários, que leem frases doadas, e é projetado para mitigar o viés em sistemas de Inteligência Artificial (IA) e democratizar a tecnologia de fala. A partir da versão 23.0, os datasets são distribuídos exclusivamente através do **Mozilla Data Collective** [1]. O dataset é ideal para o treinamento de modelos de Reconhecimento Automático de Fala (ASR), Síntese de Fala (TTS) e outras aplicações de Processamento de Linguagem Natural (PLN) [2].

## Statistics
- **Versão Mais Recente (2025)**: Common Voice 23.0 (lançada em Setembro de 2025) [3].
- **Horas Totais Gravadas**: **35.921** horas [3].
- **Horas Validadas**: **24.600** horas [3].
- **Número de Idiomas**: **286** idiomas (na versão 23.0) [3].
- **Tamanho do Arquivo (Exemplo)**: O download completo da versão 23.0 é de aproximadamente **3.51 GB** (para o segmento Single Word Target) [2].
- **Amostras**: Milhões de clipes de voz.

## Features
- **Multilinguismo Massivo**: Suporta mais de 137 idiomas, com a versão 23.0 expandindo para 286 idiomas [1] [3].
- **Dados de Fala e Texto**: Cada entrada consiste em um clipe de áudio (MP3) e o texto correspondente.
- **Metadados Demográficos**: Inclui metadados demográficos opcionais (idade, sexo, sotaque) para auxiliar no treinamento de modelos mais precisos e menos enviesados.
- **Licença Aberta**: Distribuído sob a licença **CC0** (Creative Commons Zero), permitindo o uso irrestrito e gratuito para qualquer finalidade.
- **Dados Validados**: Os dados são validados por outros voluntários, garantindo a qualidade do corpus.
- **Tipos de Fala**: Inclui fala roteirizada (Scripted Speech) e, mais recentemente, fala espontânea (Spontaneous Speech) [2].

## Use Cases
- **Reconhecimento Automático de Fala (ASR)**: Treinamento de modelos de ASR para transcrição de voz em texto.
- **Síntese de Fala (TTS)**: Criação de vozes sintéticas (embora o dataset seja primariamente para ASR, os dados de texto e áudio são úteis).
- **Processamento de Linguagem Natural (PLN)**: Pesquisa e desenvolvimento em áreas como identificação de sotaque, detecção de emoção e análise de diversidade linguística.
- **Democratização da IA**: Desenvolvimento de tecnologias de voz para idiomas com poucos recursos, combatendo o viés linguístico em sistemas comerciais [1].

## Integration
O dataset Common Voice é distribuído como um arquivo `.tar.gz` por idioma. O download é feito através do **Mozilla Data Collective** [1].

**Passos para Download:**
1.  Acesse o **Mozilla Data Collective** (URL principal).
2.  Procure por "Common Voice" e selecione o dataset desejado (ex: "Common Voice Scripted Speech 23.0").
3.  O download é geralmente iniciado após o fornecimento de um endereço de e-mail e a aceitação dos termos de uso, que incluem o compromisso de não tentar identificar os falantes.
4.  Para downloads de arquivos grandes, é recomendável usar ferramentas de linha de comando como `curl` com a opção `-C` para retomar downloads interrompidos [2].

**Estrutura do Arquivo:**
Cada arquivo `.tar.gz` contém:
-   `clips/`: Arquivos `.mp3` dos clipes de áudio.
-   Arquivos `.tsv` (tab-separated values) para diferentes partições: `train.tsv`, `dev.tsv`, `test.tsv`, `validated.tsv`, `invalidated.tsv`, `other.tsv`.
-   Cada linha do `.tsv` contém o `client_id` (anonimizado), o caminho do arquivo, a transcrição (`text`), e metadados demográficos [2].

**Uso com Bibliotecas:**
O dataset é amplamente suportado por bibliotecas de PLN, como o **Hugging Face Datasets**, onde pode ser carregado diretamente:
```python
from datasets import load_dataset

# Exemplo para a versão 13.0 (versões mais recentes podem exigir o download manual)
common_voice = load_dataset("mozilla-foundation/common_voice_13_0", "pt")
```

## URL
[https://datacollective.mozillafoundation.org/](https://datacollective.mozillafoundation.org/)
