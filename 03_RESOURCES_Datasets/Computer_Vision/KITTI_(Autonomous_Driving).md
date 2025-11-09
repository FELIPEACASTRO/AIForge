# KITTI (Autonomous Driving)

## Description
O **KITTI Vision Benchmark Suite** é um dos datasets mais influentes e amplamente utilizados para pesquisa em **visão computacional** e **direção autônoma**. Desenvolvido pelo Karlsruhe Institute of Technology (KIT) e pelo Toyota Technological Institute at Chicago (TTIC), ele fornece dados de sensores multimodais de um veículo em movimento. O veículo de coleta estava equipado com duas câmeras estéreo de alta resolução (cor e escala de cinza), um scanner a laser **Velodyne 3D** e um sistema de localização **GPS/IMU**, todos sincronizados a 10 Hz. O dataset foi capturado em áreas urbanas, rurais e rodovias na cidade de Karlsruhe, Alemanha, apresentando cenários complexos e realistas. O objetivo principal é fornecer benchmarks desafiadores para tarefas de percepção em tempo real.

## Statistics
**Imagens de Treinamento/Teste:** O benchmark de Detecção de Objetos 3D, por exemplo, consiste em 7.481 imagens de treinamento e 7.518 imagens de teste. **Objetos Anotados:** O benchmark de Detecção de Objetos 3D contém um total de 80.256 objetos rotulados. **Tamanho do Dataset:** O tamanho total do dataset (dados brutos) é substancial, com o conjunto de dados de Odometria (nuvens de pontos Velodyne) sozinho atingindo cerca de **80 GB**. **Versões:** A versão original é de 2012, com atualizações em 2015. A versão mais recente e expandida é o **KITTI-360**, que oferece maior escala e anotações semânticas de instância mais ricas.

## Features
**Multimodalidade de Sensores:** Combina imagens estéreo (cor e escala de cinza), nuvens de pontos 3D do LiDAR e dados de localização/orientação (GPS/IMU). **Cenários Realistas:** Dados coletados em condições de tráfego e ambiente variadas (cidade, rural, rodovia). **Diversidade de Benchmarks:** Suporta múltiplos benchmarks, incluindo: Estéreo, Fluxo Óptico, Odometria Visual, Detecção de Objetos 2D/3D, Rastreamento de Objetos (Tracking), Segmentação Semântica e Previsão de Profundidade. **Anotações Detalhadas:** Inclui anotações de alta qualidade para objetos 3D (carros, pedestres, ciclistas, etc.) e rótulos de rastreamento.

## Use Cases
**Direção Autônoma:** Treinamento e avaliação de sistemas de percepção para veículos autônomos. **Detecção e Rastreamento de Objetos:** Desenvolvimento de algoritmos para identificar e seguir veículos, pedestres e ciclistas em 2D e 3D. **Odometria Visual e SLAM:** Avaliação de métodos para estimativa de movimento e mapeamento simultâneo usando dados de câmera e LiDAR. **Previsão de Profundidade:** Treinamento de modelos para estimar a profundidade de uma cena a partir de imagens estéreo ou monoculares. **Segmentação Semântica:** Classificação de cada pixel da imagem ou ponto da nuvem de pontos em categorias como estrada, vegetação, veículos, etc.

## Integration
O download dos dados brutos e dos benchmarks específicos requer registro e login no site oficial. Os dados são fornecidos em formato de arquivo (PNG para imagens, binário para nuvens de pontos, texto para GPS/IMU e XML para rótulos de rastreamento). O site oficial disponibiliza um **script de download** para facilitar a obtenção dos dados brutos. Kits de desenvolvimento (devkits) em linguagens como Python e MATLAB são fornecidos para auxiliar na leitura, processamento e avaliação dos dados. O dataset também está disponível em plataformas como **TensorFlow Datasets** e **Hugging Face Datasets** para integração mais fácil em pipelines de aprendizado de máquina.

## URL
[https://www.cvlibs.net/datasets/kitti/](https://www.cvlibs.net/datasets/kitti/)
