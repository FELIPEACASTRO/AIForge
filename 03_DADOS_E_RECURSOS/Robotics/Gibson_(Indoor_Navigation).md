# Gibson (Indoor Navigation)

## Description
O **Gibson Dataset** consiste em reconstruções 3D de mais de 500 espaços internos reais (casas, escritórios, hotéis, museus, hospitais, etc.), capturados com um dispositivo Matterport. É um recurso fundamental para a pesquisa em **IA Incorporada (Embodied AI)**, especialmente para tarefas de **Navegação Interna (Indoor Navigation)**. O dataset é a base para o ambiente de simulação **iGibson**, que o estende para permitir a **Navegação Interativa**, onde os agentes podem interagir com objetos e ambientes. As cenas são fotorrealistas e de alta qualidade, mantendo a textura observada pelo sensor.

## Statistics
O **Gibson Environment Dataset** completo consiste em **572 modelos** de espaços e **1440 andares**. O tamanho total do arquivo `gibson_v2_all.tar.gz` é de aproximadamente **108GB**. Versões menores incluem a partição **4+** com **106 cenas (2.6GB)** e o **Stanford 2D-3D-Semantics** com **7 cenas (1.4GB)**. O dataset é frequentemente atualizado e integrado ao ambiente **iGibson 2.0**.

## Features
O dataset inclui mais de 500 cenas estáticas fotorrealistas de alta qualidade. Cada espaço possui metadados detalhados, como:
*   **Área**: Área total em metros quadrados.
*   **Andares**: Número de andares.
*   **Complexidade de Navegação (Navigation Complexity)**: Métrica que mede a dificuldade de navegação entre pontos arbitrários.
*   **Área de Superfície Específica (SSA - Specific Surface Area)**: Uma medida de desordem (clutter) no ambiente.
*   O dataset é compatível com o simulador iGibson, que permite interações com objetos e ambientes.

## Use Cases
*   **Treinamento de Agentes de IA**: Utilizado para treinar e avaliar agentes de IA em tarefas de Navegação Interna (Indoor Navigation) e localização.
*   **Navegação Interativa**: Em conjunto com o iGibson, é usado para pesquisa em Navegação Interativa e manipulação de objetos em ambientes realistas.
*   **Visão Computacional**: Estudos de percepção visual, reconstrução 3D e mapeamento de ambientes internos.
*   **Robótica**: Simulação de robôs móveis em ambientes complexos e fotorrealistas.

## Integration
O acesso ao dataset requer o preenchimento de um formulário de licença no site oficial do iGibson. Após a aprovação, o download pode ser realizado manualmente ou via linha de comando, utilizando o utilitário do iGibson: `python -m igibson.utils.assets_utils --download_dataset URL`. Os arquivos devem ser descompactados no caminho configurado no iGibson (padrão: `your_installation_path/igibson/data/g_dataset`). O dataset principal é fornecido como `gibson_v2_all.tar.gz`.

## URL
[https://stanfordvl.github.io/iGibson/dataset.html](https://stanfordvl.github.io/iGibson/dataset.html)
