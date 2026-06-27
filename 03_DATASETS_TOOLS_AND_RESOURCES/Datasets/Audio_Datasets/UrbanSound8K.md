# UrbanSound8K

## Description
O **UrbanSound8K** é um dataset de áudio amplamente utilizado para a classificação de sons urbanos. Ele contém **8.732 excertos de som rotulados** (com duração máxima de 4 segundos) provenientes de 10 classes distintas de sons urbanos. As classes são: ar-condicionado, buzina de carro, crianças brincando, latido de cachorro, perfuração (drilling), motor em marcha lenta, tiro, britadeira (jackhammer), sirene e música de rua. Os excertos foram extraídos de gravações de campo carregadas no site Freesound.org. O dataset é pré-organizado em **10 *folds*** (dobras) para facilitar a validação cruzada e garantir a comparabilidade dos resultados com a literatura existente. Um arquivo CSV (`UrbanSound8k.csv`) acompanha os arquivos de áudio, fornecendo metadados detalhados para cada excerto, incluindo o ID da gravação original, o tempo de início e fim do excerto, uma classificação de saliência (1=primeiro plano, 2=segundo plano) e o *fold* ao qual pertence.

## Statistics
- **Tamanho do Dataset:** Aproximadamente **6.0 GB** (Zenodo) a **7.0 GB** (Hugging Face/Kaggle) no formato compactado/bruto.
- **Amostras:** **8.732** excertos de som rotulados.
- **Classes:** **10** classes de sons urbanos (air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music).
- **Versão:** A versão original e mais citada é a de 2014. Versões mais recentes (2023-2025) são geralmente adaptações ou extensões do dataset original, como o US8K_AV.

## Features
- **Multiclasse e Rotulado:** 10 classes distintas de sons urbanos.
- **Formato de Áudio:** Arquivos WAV com taxas de amostragem e profundidades de bits variáveis (mantendo as do arquivo original do Freesound).
- **Estrutura para Validação Cruzada:** Pré-dividido em 10 *folds* (dobras) para facilitar a avaliação de modelos e garantir a reprodutibilidade.
- **Metadados Ricos:** Inclui um arquivo CSV com informações detalhadas como `fsID` (ID do Freesound), `start`, `end`, `salience` (saliência), `fold`, `classID` e `class`.
- **Duração Curta:** Excerto de som com duração máxima de 4 segundos.

## Use Cases
- **Classificação de Sons Urbanos (ESC):** O principal caso de uso, servindo como *benchmark* para o reconhecimento automático de eventos sonoros em ambientes urbanos.
- **Sistemas de Monitoramento de Ruído:** Desenvolvimento de sistemas para monitorar e analisar a poluição sonora em cidades.
- **Veículos Autônomos (AV):** Pesquisas recentes (2024-2025) utilizam o UrbanSound8K (e suas extensões, como o US8K_AV) para dar aos veículos a capacidade de "ouvir" e interpretar sons ambientais (buzinas, sirenes, britadeiras) para melhorar a segurança e a tomada de decisões.
- **Cidades Inteligentes (*Smart Cities*):** Aplicações em segurança pública (detecção de tiros ou sirenes) e gerenciamento de tráfego.
- **Pesquisa em Aprendizado Profundo (Deep Learning):** Utilizado para testar e comparar o desempenho de novas arquiteturas de redes neurais (CNN, RNN, LSTM, modelos híbridos) para processamento de áudio.

## Integration
O dataset pode ser obtido de duas maneiras principais:

1.  **Download via Navegador:** Preenchendo o formulário de download na página oficial para receber um link direto.
2.  **Download via Python (Recomendado):** Utilizando o pacote `soundata`.
    *   Instale o pacote: `pip install soundata`
    *   Use o código Python para baixar e carregar o dataset, seguindo o exemplo fornecido na documentação do `soundata`.

**Instruções de Uso:** É crucial utilizar os **10 *folds* pré-definidos** para a validação cruzada de 10 dobras, conforme recomendado pelos autores. A reordenação dos dados ou o uso de menos *folds* pode levar a resultados inflacionados e não comparáveis com a literatura.

**Agradecimento:** O dataset deve ser citado em pesquisas acadêmicas usando a referência do artigo original: Salamon, J., Jacoby, C. & Bello, J.P. A dataset and taxonomy for urban sound research. *Proceedings of the 22nd ACM International Conference on Multimedia* (2014).

## URL
[https://urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html)
