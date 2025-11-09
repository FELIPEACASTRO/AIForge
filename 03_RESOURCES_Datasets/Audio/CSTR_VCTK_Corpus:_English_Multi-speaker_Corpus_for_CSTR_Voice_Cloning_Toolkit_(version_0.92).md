# CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)

## Description
O **CSTR VCTK Corpus** (Centre for Speech Technology Voice Cloning Toolkit) é um conjunto de dados de fala multilíngue em inglês, projetado primariamente para pesquisa em **síntese de fala (Text-to-Speech - TTS)**, especialmente para sistemas de síntese adaptativa de falante. A versão mais comum (v0.92) inclui dados de fala proferidos por **110 falantes nativos de inglês** com uma variedade de sotaques regionais. Cada falante lê cerca de 400 frases, selecionadas de um jornal (The Herald, Glasgow), além da "Rainbow Passage" e um parágrafo de eliciação de sotaque. O corpus é amplamente utilizado para treinar modelos de TTS baseados em HMM (Hidden Markov Model) e, mais recentemente, sistemas de síntese de fala multi-falante baseados em Redes Neurais Profundas (DNNs) e modelagem de forma de onda neural, como o WaveNet. O corpus é notável por sua alta qualidade de gravação, realizada em uma câmara hemi-anecoica da Universidade de Edimburgo, usando microfones de alta fidelidade (DPA 4035 e Sennheiser MKH 800), com taxa de amostragem de 96kHz e 24 bits, posteriormente convertidos para 48kHz e 16 bits.

## Statistics
- **Tamanho Total:** 10.94 GB (arquivo principal).
- **Falantes:** 110 falantes nativos de inglês (109 com transcrições completas, `p315` perdeu o arquivo de texto).
- **Amostras:** Cada falante lê aproximadamente 400 frases. O total de clipes de áudio é de cerca de 44.000.
- **Duração Total:** Aproximadamente 44 horas de fala.
- **Taxa de Amostragem:** 48 kHz (originalmente gravado em 96 kHz).
- **Versão Principal:** 0.92 (disponível desde 2019-11-13).

## Features
- **Multi-falante e Multi-sotaque:** Contém 110 falantes nativos de inglês com diversos sotaques regionais, tornando-o ideal para modelos de TTS adaptativos e multi-falantes.
- **Alta Qualidade de Gravação:** Gravado em câmara hemi-anecoica com microfones profissionais (DPA 4035 e Sennheiser MKH 800) a 96kHz/24 bits, posteriormente downsampled para 48kHz/16 bits.
- **Conteúdo Variado:** As frases incluem textos de jornal, a "Rainbow Passage" (para análise fonética) e um parágrafo de eliciação de sotaque.
- **Foco em Síntese de Fala:** Originalmente destinado a sistemas de TTS baseados em HMM e, atualmente, crucial para o desenvolvimento de modelos neurais de TTS (como VITS e WaveNet).

## Use Cases
- **Síntese de Fala (Text-to-Speech - TTS):** Treinamento de modelos de TTS de alta qualidade, incluindo sistemas neurais como WaveNet, Tacotron e VITS.
- **Clonagem de Voz:** Desenvolvimento de sistemas de clonagem de voz e síntese adaptativa de falante.
- **Reconhecimento de Fala:** Embora não seja o foco principal, pode ser usado para treinamento e avaliação de modelos de reconhecimento de fala multi-falante.
- **Análise de Sotaque:** Pesquisa em variação fonética e sotaques regionais do inglês.
- **Aprimoramento de Fala:** Utilizado como base para a criação de datasets derivados para aprimoramento de fala (ex: VCTK-RVA para atributos de voz).

## Integration
O dataset VCTK (versão 0.92) está disponível para download no repositório Edinburgh DataShare. O download principal é um arquivo de **10.94 GB** que contém os arquivos de áudio e texto.

**Passos para Integração:**
1. **Acesso:** Navegue até a página do recurso no Edinburgh DataShare (URL principal fornecida).
2. **Download:** Clique no link de download do "Main file including audio and text files (10.94Gb)".
3. **Estrutura:** O corpus geralmente é organizado em pastas para cada falante (`p225`, `p226`, etc.), contendo os arquivos de áudio (`.wav`) e os arquivos de transcrição correspondentes (`.txt`).
4. **Uso:** Para uso em projetos de aprendizado de máquina, é comum utilizar bibliotecas como `torchaudio` ou `tensorflow_datasets` que podem oferecer wrappers para o VCTK, ou processar manualmente os arquivos de áudio e texto para criar pares de treinamento. Por exemplo, o `tensorflow_datasets` oferece uma versão do VCTK pronta para uso.
5. **Citação:** É obrigatório citar o trabalho original ao utilizar o corpus.

## URL
[https://datashare.ed.ac.uk/handle/10283/3443](https://datashare.ed.ac.uk/handle/10283/3443)
