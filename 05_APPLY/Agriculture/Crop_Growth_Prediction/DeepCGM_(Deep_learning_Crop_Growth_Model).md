# DeepCGM (Deep learning Crop Growth Model)

## Description

O **DeepCGM (Deep learning Crop Growth Model)** é um modelo de crescimento de culturas baseado em aprendizado profundo que incorpora restrições guiadas por conhecimento para garantir simulações fisicamente plausíveis. Ele aborda as limitações dos modelos tradicionais baseados em processos (como o ORYZA2000), que sofrem com a simplificação e a dificuldade na estimação de parâmetros, e dos modelos de aprendizado de máquina clássicos, que são criticados por serem "caixas-pretas" e exigirem grandes volumes de dados. O DeepCGM utiliza uma arquitetura de conservação de massa e restrições fisiológicas da cultura para operar com dados esparsos de séries temporais.

## Statistics

**Melhoria na Precisão:** Supera o modelo tradicional baseado em processos ORYZA2000, com a precisão geral (erro quadrático médio normalizado ponderado) em todas as variáveis melhorando em **8,3% (2019)** e **16,9% (2018)**. **Citações:** O artigo associado, "Knowledge-guided machine learning with multivariate sparse data for crop growth modelling" (J. Han et al., 2025), já possui **2 citações** (em 2025), indicando relevância recente na comunidade científica. **Publicação:** O trabalho foi publicado na revista *Field Crops Research* em 2025.

## Features

**Arquitetura de Conservação de Massa:** Adere aos princípios de crescimento da cultura, como a conservação de massa, para garantir previsões fisicamente realistas. **Restrições Guiadas por Conhecimento:** Inclui restrições de fisiologia da cultura e convergência do modelo, permitindo previsões precisas mesmo com dados esparsos. **Previsão Multivariável:** Simula múltiplas variáveis de crescimento da cultura (por exemplo, biomassa, área foliar) em uma única estrutura. **Código Aberto:** O código está disponível no GitHub, facilitando a pesquisa e a implementação.

## Use Cases

**Simulação de Crescimento de Culturas:** Simulação precisa e fisicamente plausível de variáveis de crescimento como Índice de Área Foliar (PAI), biomassa de órgãos individuais (folha, caule, grão) e biomassa total acima do solo (WAGT). **Modelagem com Dados Escassos:** Ideal para cenários agrícolas onde a coleta de dados de séries temporais é esparsa ou incompleta. **Substituição de Modelos Processuais:** Serve como uma alternativa mais precisa e robusta aos modelos de crescimento de culturas baseados em processos tradicionais, como o ORYZA2000.

## Integration

O modelo é implementado em Python e pode ser treinado usando o script `train.py` no repositório do GitHub. A instalação é feita via `conda` e `pip` a partir do arquivo `requirements.txt`.

**Exemplo de Treinamento:**
```shell
git clone https://github.com/WUR-AI/DeepCGM.git
cd DeepCGM
conda create -n DeepCGM python==3.10.16
conda activate DeepCGM
pip install -r ./requirements.txt
python train.py --model DeepCGM --target spa --input_mask 1 --convergence_loss 1 --tra_year 2018
```
O script permite especificar o tipo de modelo (`NaiveLSTM`, `MCLSTM`, `DeepCGM`), o rótulo de treinamento (`spa` para dados esparsos), e habilitar o uso de máscara de entrada e a perda de convergência.

## URL

https://github.com/WUR-AI/DeepCGM