# Gene Ontology (GO) Dataset

## Description
O **Gene Ontology (GO)** é um recurso fundamental na bioinformática, fornecendo uma estrutura hierárquica e controlada de vocabulários para descrever as funções de genes e produtos proteicos em qualquer organismo. O GO é composto por três ontologias principais: **Função Molecular (Molecular Function - MF)**, que descreve as atividades moleculares; **Processo Biológico (Biological Process - BP)**, que descreve as séries de eventos biológicos; e **Componente Celular (Cellular Component - CC)**, que descreve os locais onde os produtos gênicos atuam. O GO Consortium mantém e atualiza continuamente esta base de conhecimento, que é essencial para a anotação funcional de genomas e para a análise de dados de alto rendimento em biologia.

## Statistics
**Versão Mais Recente (2025-10-10):**
*   **Termos Válidos (Ontologia):** 39.354
*   **Anotações Totais:** 9.281.704
*   **Produtos Gênicos Anotados:** 1.601.555
*   **Espécies Anotadas:** 5.495
*   **Publicações Científicas Anotadas:** 187.286
*   **Tamanho do Arquivo:** O tamanho varia dependendo do formato e do conjunto de anotações (GAF, GPAD/GPI) e da ontologia (OBO, OWL). O arquivo de ontologia principal é de alguns MBs, mas os arquivos de anotação podem ser de centenas de MBs a vários GBs.

## Features
**Estrutura Hierárquica:** Organizado como um grafo acíclico dirigido (DAG), permitindo que termos mais específicos se liguem a termos mais gerais. **Três Ontologias:** Cobre Funções Moleculares, Processos Biológicos e Componentes Celulares. **Análise de Enriquecimento:** Permite identificar termos GO significativamente sobrerrepresentados em um conjunto de genes, indicando as funções biológicas mais relevantes. **Análise de Anotação:** Associa termos GO a produtos gênicos com base em evidências experimentais ou computacionais. **Padrão Aberto:** Os arquivos de ontologia (OBO, OWL) e anotação (GAF, GPAD/GPI) são abertos e amplamente utilizados.

## Use Cases
**Análise de Dados de Alto Rendimento:** Interpretação de resultados de experimentos de transcriptômica (RNA-seq), proteômica e genômica. **Predição de Função Proteica:** Treinamento de modelos de *machine learning* e *deep learning* (como DeepGOPlus) para prever a função de proteínas recém-descobertas. **Pesquisa em Doenças:** Identificação de vias biológicas e funções celulares afetadas por mutações ou alterações de expressão gênica em doenças. **Genômica Comparativa:** Comparação de funções gênicas entre diferentes espécies.

## Integration
O dataset GO é acessível através de diversos métodos. Os arquivos de ontologia (OBO, OWL) e anotações (GAF, GPAD/GPI) são disponibilizados para download.
1.  **Download Direto:** Os arquivos mais recentes podem ser baixados do site oficial (por exemplo, `http://current.geneontology.org/`).
2.  **Arquivos Históricos:** Versões mensais arquivadas estão disponíveis no Zenodo (e.g., [Zenodo - record 1205166](https://zenodo.org/record/1205166)).
3.  **Ferramentas:** O GO pode ser integrado em ferramentas de bioinformática como **AmiGO**, **QuickGO**, e pacotes de análise de enriquecimento (e.g., **GOseq**, **topGO** no R/Bioconductor).
4.  **Uso em IA:** Para projetos de IA, os termos GO são frequentemente usados como *labels* para tarefas de classificação de função proteica (e.g., em modelos como DeepGOPlus), ou para enriquecer *features* em modelos de predição. Recomenda-se o uso dos arquivos GAF (Gene Association File) ou GPAD/GPI para anotações.

## URL
[http://geneontology.org/](http://geneontology.org/)
