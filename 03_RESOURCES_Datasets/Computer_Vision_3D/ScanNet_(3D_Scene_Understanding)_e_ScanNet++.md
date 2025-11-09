# ScanNet (3D Scene Understanding) e ScanNet++

## Description
O ScanNet é um dataset de vídeo RGB-D amplamente utilizado para a compreensão de cenas 3D internas. A versão original (v2) contém 2,5 milhões de visualizações em mais de 1500 cenas, anotadas com poses de câmera 3D, reconstruções de superfície e segmentações semânticas em nível de instância. A versão mais recente e de alta fidelidade, **ScanNet++** (ICCV 2023 Oral), expande significativamente o dataset, oferecendo mais de **1000 cenas** internas com digitalizações a laser de resolução submilimétrica, imagens DSLR de 33 megapixels registradas e streams RGB-D de dispositivos commodity (iPhone). O ScanNet++ foca em anotações semânticas de cauda longa e ambíguas para aprimorar os métodos de compreensão semântica e também suporta benchmarks de síntese de novas visualizações (Novel View Synthesis - NVS) em configurações de alta qualidade e commodity. O dataset é fundamental para o avanço da pesquisa em visão computacional e robótica.

## Statistics
**ScanNet v2 (Original):**
*   **Cenas:** Mais de 1500.
*   **Visualizações:** 2,5 milhões de quadros RGB-D.
*   **Versão:** v2 (lançada em 2018).

**ScanNet++ (Versão mais recente):**
*   **Cenas:** Mais de 1000 (v2, Dezembro de 2024).
*   **Resolução:** Digitalizações a laser de resolução submilimétrica.
*   **Imagens:** Imagens DSLR de 33 megapixels.
*   **Publicação:** ICCV 2023 Oral.

## Features
**ScanNet v2:**
*   Vídeos RGB-D.
*   2,5 milhões de visualizações.
*   Mais de 1500 cenas internas.
*   Anotações de pose de câmera 3D, reconstruções de superfície e segmentação semântica em nível de instância.

**ScanNet++ (v2, Dezembro de 2024):**
*   Mais de **1000 cenas** internas.
*   Digitalizações a laser de **resolução submilimétrica**.
*   Imagens DSLR de **33 megapixels** registradas.
*   Streams RGB-D de dispositivos commodity (iPhone).
*   Anotações semânticas de cauda longa e ambíguas.
*   Suporte a benchmarks de Síntese de Novas Visualizações (NVS).
*   **ScanNet200 Benchmark** para segmentação semântica com 200 categorias de classes.

## Use Cases
*   **Compreensão de Cenas 3D:** Classificação de objetos 3D, rotulagem semântica de voxels e segmentação semântica e de instância 3D.
*   **Robótica:** Navegação e interação de robôs em ambientes internos.
*   **Realidade Aumentada/Virtual (AR/VR):** Reconstrução de ambientes internos de alta fidelidade.
*   **Síntese de Novas Visualizações (NVS):** Geração de novas perspectivas de cenas internas.
*   **Aprendizado por Contraste:** Benchmarks de compreensão de cenas 3D com eficiência de dados (ScanNet-LA).

## Integration
O acesso ao dataset ScanNet original (v2) e ao ScanNet++ requer a aceitação dos Termos de Uso e a obtenção de uma licença não comercial.

1.  **ScanNet (v2):** O código e os dados estão disponíveis no repositório oficial do GitHub. É necessário concordar com os termos de uso e seguir as instruções no site principal para obter acesso aos dados.
    *   **URL de Código e Dados:** `https://github.com/ScanNet/ScanNet`
2.  **ScanNet++:** É necessário criar uma conta, fazer login e enviar uma solicitação de acesso no site oficial. Após a aprovação, um token personalizado e um novo script de download são fornecidos para acessar os dados.
    *   **URL de Download:** `https://scannetpp.mlsg.cit.tum.de/scannetpp/` (Requer registro e aprovação)

## URL
[http://www.scan-net.org/](http://www.scan-net.org/)
