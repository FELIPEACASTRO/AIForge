# nuScenes (Autonomous Driving)

## Description
O nuScenes é um **dataset público e em larga escala** para direção autônoma, desenvolvido pela Motional (anteriormente nuTonomy). Ele foi projetado para permitir que pesquisadores estudem situações desafiadoras de direção urbana usando o conjunto completo de sensores de um veículo autônomo real. O dataset é composto por 1000 cenas de 20 segundos de duração, coletadas em Boston e Singapura, cidades conhecidas por seu tráfego denso e situações de direção complexas. O nuScenes é o primeiro dataset em larga escala a fornecer dados de um conjunto completo de sensores (6 câmeras, 1 LIDAR, 5 RADAR, GPS, IMU) e inclui anotações de caixas delimitadoras 3D precisas para 23 classes de objetos, além de atributos como visibilidade, atividade e pose. Versões estendidas, como o **nuScenes-lidarseg** (segmentação semântica de pontos LIDAR) e o **Panoptic nuScenes** (segmentação e rastreamento panóptico de nuvens de pontos), foram lançadas para tarefas mais avançadas.

## Statistics
**Cenas:** 1000 cenas de 20 segundos cada (aproximadamente 5.5 horas de direção). **Imagens:** 1.400.000 imagens de câmera. **Varreduras LIDAR:** 390.000 varreduras LIDAR. **Caixas Delimitadoras 3D:** 1.4M de caixas delimitadoras 3D anotadas em 40k *keyframes*. **Pontos LIDAR Anotados (nuScenes-lidarseg):** 1.4 bilhão de pontos anotados em 40.000 nuvens de pontos. **Versão Principal:** nuScenes v1.0 (lançada em Março de 2019). **Versões do Devkit:** A versão mais recente do devkit é a v1.2.0 (Agosto de 2025), compatível com Python 3.9 e 3.12.

## Features
**Conjunto Completo de Sensores:** 1x LIDAR (Velodyne HDL32E), 5x RADAR (Continental ARS 408-21), 6x Câmeras (Basler acA1600-60gc), IMU e GPS. **Anotações 3D:** 1.4M de caixas delimitadoras 3D anotadas manualmente para 23 classes de objetos. **Diversidade Geográfica:** Dados coletados em duas cidades distintas (Boston e Singapura), permitindo o estudo da generalização de algoritmos em diferentes condições de tráfego (mão inglesa vs. mão americana), clima e ambientes. **Expansões:** Inclui expansões como **nuScenes-lidarseg** (1.4 bilhão de pontos LIDAR anotados com 32 classes semânticas) e **Panoptic nuScenes** (segmentação panóptica de nuvens de pontos). **Dados de Baixo Nível:** Expansão de barramento CAN (CAN bus) com dados de veículo de baixo nível (velocidade da roda, ângulo de direção, etc.).

## Use Cases
**Detecção e Rastreamento de Objetos 3D:** Principal aplicação para o desenvolvimento de algoritmos de percepção. **Segmentação Semântica de Nuvens de Pontos:** Utilizando a expansão nuScenes-lidarseg. **Segmentação e Rastreamento Panóptico:** Com o Panoptic nuScenes. **Previsão de Comportamento:** Desafios de previsão de trajetória de agentes. **Fusão de Sensores:** Desenvolvimento de modelos que combinam dados de câmeras, LIDAR e RADAR. **Localização e Mapeamento:** Uso dos dados de GPS/IMU e mapas HD.

## Integration
O dataset nuScenes é disponibilizado gratuitamente para uso estritamente **não comercial**. Para fazer o download, é necessário **registrar-se e fazer login** no site oficial (https://www.nuscenes.org/nuscenes#download) e concordar com os Termos de Uso. O download é feito por meio de múltiplos arquivos compactados que devem ser descompactados em uma estrutura de pastas específica (e.g., `/data/sets/nuscenes`). O kit de desenvolvimento (**nuscenes-devkit**) é essencial para trabalhar com o dataset e pode ser instalado via pip: `pip install nuscenes-devkit`. O devkit fornece ferramentas para visualização, manipulação de dados e avaliação de modelos. O dataset também está disponível no **Registry of Open Data on AWS**.

## URL
[https://www.nuscenes.org/](https://www.nuscenes.org/)
