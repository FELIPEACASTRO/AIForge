# ESC-50 (Environmental Sound Classification)

## Description
O ESC-50 (Environmental Sound Classification) é um conjunto de dados rotulado composto por 2.000 gravações de áudio ambiental, com 5 segundos de duração cada, adequado para a avaliação de métodos de classificação de sons ambientais. O dataset é balanceado, contendo 50 classes distintas (40 clipes por classe), agrupadas em 5 categorias principais: Animais, Fenômenos Naturais, Materiais, Sons Humanos Não-Vocais e Sons Urbanos Domésticos. As gravações foram extraídas do banco de dados **Freesound** e pré-processadas para garantir a qualidade e a uniformidade. É amplamente utilizado como um benchmark padrão na pesquisa de classificação de áudio.

## Statistics
- **Tamanho Total:** Aproximadamente 500 MB (arquivos WAV).
- **Amostras:** 2.000 clipes de áudio.
- **Duração dos Clipes:** 5 segundos cada.
- **Classes:** 50 classes de sons ambientais.
- **Amostras por Classe:** 40 clipes por classe (dataset balanceado).
- **Taxa de Amostragem:** 44.1 kHz.
- **Versão:** A versão original foi introduzida em 2015. Versões derivadas e aprimoradas, como o **ESC50Mix** (2023), que adiciona misturas de sons, continuam a ser desenvolvidas com base no ESC-50.

## Features
- **50 Classes de Sons Ambientais:** Inclui sons como latido de cachorro, chuva, motor de carro, tossir, etc.
- **Estrutura de Classes:** As 50 classes são agrupadas em 5 categorias principais para facilitar a análise e a classificação hierárquica.
- **Gravações de 5 Segundos:** Todos os clipes de áudio têm uma duração fixa de 5 segundos.
- **Áudio em Formato WAV:** Os arquivos de áudio são fornecidos no formato WAV, com taxa de amostragem de 44.1 kHz.
- **Divisão em 5 Folds:** O dataset é pré-dividido em 5 folds para validação cruzada, conforme sugerido pelos autores, garantindo uma avaliação de modelo justa e reproduzível.

## Use Cases
- **Classificação de Sons Ambientais (ESC):** Tarefa primária para a qual o dataset foi projetado.
- **Detecção de Eventos Sonoros (SED):** Embora seja um dataset de classificação, é frequentemente usado para pré-treinamento ou como base para tarefas de detecção.
- **Sistemas de Monitoramento de Áudio:** Aplicações em segurança, monitoramento de vida selvagem e sistemas de casa inteligente (ex: detecção de quebra de vidro, alarme de fumaça).
- **Pesquisa em Aprendizado Profundo (Deep Learning):** Benchmark para o desenvolvimento e teste de novas arquiteturas de redes neurais convolucionais (CNNs) e transformadores para processamento de áudio.
- **Transfer Learning:** Uso do dataset para pré-treinar modelos que serão ajustados para tarefas de áudio mais específicas.

## Integration
O dataset pode ser baixado diretamente do repositório oficial no GitHub ou de plataformas como Kaggle e Hugging Face.

**Download e Uso:**
1. **GitHub (Fonte Principal):** Baixe o arquivo ZIP do repositório oficial.
   - `git clone https://github.com/karolpiczak/ESC-50.git`
2. **Kaggle:** Disponível para download e uso em notebooks Kaggle.
3. **Hugging Face Datasets:** Pode ser carregado diretamente em projetos Python usando a biblioteca `datasets`.
   - `from datasets import load_dataset`
   - `dataset = load_dataset("ashraq/esc50")`

O dataset é composto por uma pasta de áudios (`audio`) e um arquivo CSV de metadados (`meta/esc50.csv`) que contém o nome do arquivo, a classe, o ID do fold de validação cruzada e a categoria principal. É recomendado usar a divisão de 5 folds fornecida para treinamento e teste.

## URL
[https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)
