# Knowledge Distillation for Edge Medical Devices

## Description

Pesquisa abrangente sobre a aplicação de **Destilação de Conhecimento (Knowledge Distillation - KD)** para otimizar modelos de Inteligência Artificial para **Dispositivos Médicos de Borda (Edge Medical Devices)**. Foram identificados três recursos-chave recentes (2023-2025) que demonstram a eficácia do KD na compressão de modelos de diagnóstico e segmentação de imagens médicas, mantendo alta acurácia e reduzindo a latência para implantação em ambientes com recursos limitados. Os recursos incluem um framework de KD para diagnóstico de COVID-19 e Malária, uma avaliação comparativa de técnicas de compressão (KD vs. Quantização) e um framework avançado (KnowSAM) que utiliza KD para segmentação de imagens médicas semi-supervisionada, aproveitando o poder do Segment Anything Model (SAM).

## Statistics

**Compressão:** Até 18.4% (COVID-19) e 15% (Malária) do modelo original. **Aceleração:** 6.14x (COVID-19) e 5.86x (Malária). **Queda de Desempenho:** Apenas 0.9% (COVID-19) e 1.2% (Malária). **Latência:** MobileNet v3 destilado alcançou 44,8 milissegundos por inferência. **Redução de Memória:** 50 a 90 vezes com quantização de 16 bits.

## Features

Compressão de modelo, aceleração de inferência, alta retenção de acurácia, implantação em dispositivos de borda/fog, avaliação comparativa de técnicas de compressão, Destilação de Conhecimento Induzida por SAM (SKD), Estratégia de Prompt Aprendível (LPS), Co-treinamento Multi-visão (MC).

## Use Cases

Diagnóstico automático de COVID-19 e Malária em tempo real em dispositivos com recursos limitados. Implementação de modelos de IA em sistemas embarcados e dispositivos móveis. Segmentação de imagens médicas (por exemplo, órgãos, lesões) em cenários com dados rotulados limitados.

## Integration

Os recursos incluem um framework proposto para desenvolvimento de modelos DNN destilados, validação de arquiteturas de baixa latência como MobileNet v3 destilado, e o anúncio de código a ser liberado para o framework KnowSAM. O KD é apresentado como uma técnica superior a outras formas de compressão para manter o desempenho.

## URL

Múltiplas fontes (ver detalhes no JSON)