# CDTM (Cross-Domain Transfer Module) + SFT (Staged Fine-Tuning)

## Description

O **Módulo de Transferência de Domínio Cruzado (CDTM)** e a estratégia de **Fine-Tuning em Estágios (SFT)** são uma abordagem de Transfer Learning proposta para aprimorar a análise de imagens médicas, superando a limitação de dados escassos no domínio médico. O método aproveita modelos pré-treinados em grandes conjuntos de dados de imagens naturais (como ImageNet-21K, LAION-400M e LAION-2B) e os adapta eficientemente para o domínio médico. O CDTM atua como um módulo de adaptação de recursos, transferindo características do domínio de visão natural para o domínio de imagens médicas. O SFT, por sua vez, é uma estratégia de ajuste fino em duas etapas que otimiza o desempenho do modelo. O trabalho foi publicado no *IEEE Journal of Biomedical and Health Informatics* em 2023/2024.

## Statistics

O método **ConvNeXt-BCDTM-SFT** alcançou resultados de ponta (*state-of-the-art*) no dataset **BreakHis** (classificação de câncer de mama histopatológico) no momento da publicação. **Métricas de Desempenho (BreakHis):** * Acurácia: 99.80% (em comparação com 99.40% do modelo anterior mais forte, o BACH). * F1-Score: 99.80% (estimado a partir da imagem de resultados). **Datasets Utilizados:** * **Domínio Fonte (Natural):** ImageNet-21K, LAION-400M, LAION-2B. * **Domínio Alvo (Médico):** BreakHis (Histopatologia de Câncer de Mama), HCRF (Histopatologia Gástrica). **Citações:** 9 (Google Scholar, em Nov 2025).

## Features

**Módulo de Transferência de Domínio Cruzado (CDTM):** Componente que facilita a transferência de recursos do domínio de imagens naturais para o domínio médico, adaptando o modelo pré-treinado. **Estratégia de Fine-Tuning em Estágios (SFT):** Processo de ajuste fino em duas etapas (congelamento do *backbone* seguido de ajuste fino completo) que otimiza o desempenho e a eficiência. **Compatibilidade com Modelos de Visão:** Demonstração de eficácia com arquiteturas baseadas em CNN (ConvNeXt) e Transformer (ViT). **Aproveitamento de Grandes Dados Naturais:** Utiliza o conhecimento de modelos pré-treinados em datasets massivos como LAION-2B e ImageNet-21K.

## Use Cases

**Classificação de Imagens Histopatológicas:** Detecção e classificação de câncer de mama (BreakHis) e análise de histopatologia gástrica (HCRF). **Diagnóstico Auxiliado por Computador (CAD):** Criação de sistemas de CAD de alto desempenho em domínios com escassez de dados. **Adaptação Rápida de Modelos:** Permite que modelos pré-treinados em grandes datasets de imagens naturais sejam rapidamente adaptados para tarefas médicas específicas com desempenho superior. **Redução da Necessidade de Grandes Datasets Médicos:** Mitiga o desafio de coletar e anotar grandes volumes de dados médicos.

## Integration

A implementação está disponível no GitHub e utiliza o framework PyTorch. O processo de treinamento envolve: 1. **Preparação de Dados:** Download dos datasets médicos (e.g., BreakHis, HCRF) e criação de arquivos CSV para organização. 2. **Download de Pesos:** Utilização de pesos pré-treinados de modelos como ViT e ConvNeXt da biblioteca `timm`. 3. **Treinamento:** Execução de scripts Python com argumentos para selecionar o modo do modelo (`vit` ou `conv`), o modo de fine-tuning (`linear`, `full`, `frt`), e a configuração do CDTM/SFT. O SFT requer uma etapa inicial de congelamento do *backbone* seguida de uma segunda etapa de ajuste fino com os pesos da primeira etapa. **Exemplo de Comando (ViT com SFT no BreakHis):** `python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode vit --finetune-mode frt --csv-dir malignant_all_5fold.csv --config-name 'config_clip_vit' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 5` (Este comando inicia a primeira etapa do SFT).

## URL

https://github.com/qklee-lz/CDTM