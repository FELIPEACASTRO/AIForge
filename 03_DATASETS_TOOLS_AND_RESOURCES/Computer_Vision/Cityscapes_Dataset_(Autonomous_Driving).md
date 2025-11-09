# Cityscapes Dataset (Autonomous Driving)

## Description
O **Cityscapes Dataset** é um dataset em larga escala focado na **compreensão semântica de cenas urbanas de rua**, essencial para o desenvolvimento de sistemas de direção autônoma. Ele contém um conjunto diversificado de sequências de vídeo estéreo gravadas em 50 cidades diferentes. O dataset fornece anotações densas e detalhadas em nível de pixel e instância para 30 classes semânticas, permitindo o treinamento e avaliação de algoritmos de visão computacional para tarefas complexas de segmentação e detecção em ambientes urbanos.

## Statistics
*   **Volume:** 5.000 imagens com anotações finas (pixel-level) e 20.000 imagens com anotações grosseiras (fracamente anotadas).
*   **Cidades:** 50 cidades diferentes.
*   **Classes:** 30 classes semânticas.
*   **Extensão 3D:** Cityscapes 3D (lançado em Outubro de 2020) adiciona anotações de *bounding box* 3D para veículos.
*   **Resolução:** Imagens de alta resolução (tipicamente 1024x2048).

## Features
*   **Anotações Poligonais:** Segmentação semântica densa e segmentação de instância para veículos e pessoas.
*   **Complexidade:** 30 classes semânticas detalhadas (ex: estrada, calçada, carro, pessoa, vegetação, céu).
*   **Diversidade:** Cenas gravadas em 50 cidades, em diferentes meses (primavera, verão, outono), durante o dia e em condições climáticas boas/médias.
*   **Metadados Ricos:** Inclui *frames* de vídeo precedentes e seguintes, visões estéreo direitas correspondentes, coordenadas GPS e dados de ego-movimento.
*   **Extensões:** Possui a extensão **Cityscapes 3D** com anotações de *bounding box* 3D para veículos.

## Use Cases
*   **Segmentação Semântica:** Treinamento e avaliação de modelos para classificar cada pixel em uma imagem (ex: identificar estrada, calçada, edifícios).
*   **Segmentação de Instância:** Identificação e segmentação de objetos individuais (ex: carros, pessoas).
*   **Visão para Veículos Autônomos:** Componente fundamental para a percepção ambiental em sistemas de direção autônoma.
*   **Pesquisa em Visão Computacional:** Desenvolvimento de novos algoritmos de *Deep Learning* para compreensão de cenas urbanas.
*   **Detecção de Objetos 3D:** Com a extensão Cityscapes 3D, é utilizado para tarefas de detecção de veículos em 3D.

## Integration
O dataset é disponibilizado gratuitamente para fins não comerciais (pesquisa acadêmica, ensino, publicações científicas, experimentação pessoal).
1.  **Registro:** É necessário se registrar no site oficial (https://www.cityscapes-dataset.com/login/) para obter acesso aos dados.
2.  **Download:** Após o registro e aprovação, o download dos dados pode ser realizado através da seção de *Download* do site.
3.  **Ferramentas:** O dataset é acompanhado por um *toolbox* em Python (`cityscapesscripts`) para inspeção, preparação e avaliação, que pode ser instalado via `pip`: `python -m pip install cityscapesscripts[gui]`.
4.  **Uso:** O dataset é tipicamente utilizado com frameworks de *Deep Learning* (como PyTorch ou TensorFlow) para tarefas de segmentação semântica e de instância.

## URL
[https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)
