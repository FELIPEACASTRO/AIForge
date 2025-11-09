# The Rosario Dataset v2: Multi-modal dataset for agricultural robotics

## Description

O **Rosario Dataset v2** é um conjunto de dados multimodal coletado em um campo de soja, projetado especificamente para robótica agrícola. Ele abrange mais de duas horas de dados gravados de múltiplos sensores, incluindo câmera estéreo infravermelha, câmera colorida, acelerômetro, giroscópio, magnetômetro, GNSS (Single-Point Positioning, Real-Time Kinematic e Post-Processed Kinematic) e odometria de roda. O dataset foi criado para capturar desafios inerentes a ambientes agrícolas, como variações na iluminação natural, desfoque de movimento, terreno irregular e sequências longas com aliasing perceptual. É um recurso crucial para o desenvolvimento e benchmarking de algoritmos avançados de localização, mapeamento, percepção e navegação em robôs agrícolas. O suporte para ROS1 e ROS2 facilita a integração.

## Statistics

**Duração Total:** Mais de 2 horas de dados gravados. **Sequências:** 6 sequências separadas (3 de 22/Dez/2023 e 3 de 26/Dez/2023). **Distância Total Percorrida:** Aproximadamente 7.338 metros. **Sensores:** Câmera estéreo IR (1280x720, 15Hz), Câmera Colorida (1280x720, 15Hz), IMUs (200Hz), GNSS (5Hz), Odometria de Roda (10Hz). **Cultura:** Soja.

## Features

Multimodal (visão, inercial, GNSS, odometria); Sincronização de hardware de sensores; Ground-truth 6-DOF; Sequências longas com loops; Dados de campo de soja; Desafios de ambientes agrícolas (iluminação, terreno).

## Use Cases

Desenvolvimento e benchmarking de algoritmos de **SLAM (Simultaneous Localization and Mapping)** multimodal em ambientes agrícolas. Teste de sistemas de **localização, mapeamento, percepção e navegação** para robôs agrícolas autônomos. Pesquisa em **visão computacional** para detecção e rastreamento de culturas em condições adversas.

## Integration

O dataset é disponibilizado em formato **Rosbag** (ROS1 e ROS2 suportados). Os utilitários para trabalhar com o dataset estão disponíveis em um repositório GitHub. O acesso aos dados é feito através de links de download diretos para cada uma das seis sequências gravadas. A documentação fornece informações detalhadas sobre a calibração dos sensores e a estrutura dos dados.

## URL

https://cifasis.github.io/rosariov2/