# CelebA (Large-scale CelebFaces Attributes Dataset)

## Description
O **CelebFaces Attributes Dataset (CelebA)** é um conjunto de dados de grande escala para atributos faciais, contendo mais de **200.000** imagens de celebridades. Cada imagem é anotada com **40 atributos binários** (como "sorrindo", "cabelo loiro", "óculos") e **5 localizações de pontos de referência facial (landmarks)**. O dataset é notável por sua grande diversidade, cobrindo variações significativas de pose e desordem de fundo. É amplamente utilizado na pesquisa de visão computacional para tarefas relacionadas ao rosto. A fonte primária é o MMLAB da Universidade Chinesa de Hong Kong. Embora o dataset original seja de 2015, ele continua sendo uma base fundamental, com versões relacionadas mais recentes como o CelebA-HQ (alta qualidade) e o Multi-Modal-CelebA-HQ (com descrições textuais).

## Statistics
- **Imagens**: 202.599 imagens de faces.
- **Identidades**: 10.177 identidades únicas.
- **Anotações**: 40 atributos binários e 5 landmarks por imagem.
- **Versões**: A versão original é de 2015. Versões relacionadas e de alta qualidade incluem **CelebA-HQ** (30.000 imagens de alta resolução) e **Multi-Modal-CelebA-HQ** (30.000 imagens com descrições textuais).
- **Tamanho**: O tamanho total do dataset (imagens e anotações) é de aproximadamente 1.6 GB (para a versão alinhada e recortada) ou mais para a versão "in-the-wild".

## Features
- **Atributos Ricos**: 40 atributos binários por imagem, permitindo o treinamento de modelos de reconhecimento de atributos faciais.
- **Localização de Landmarks**: 5 pontos de referência facial (olhos, nariz, boca) anotados para cada imagem.
- **Grande Escala**: Mais de 200.000 imagens e 10.000 identidades únicas.
- **Diversidade**: Variações significativas de pose, expressão, iluminação e fundo.
- **Imagens Alinhadas e Recortadas**: Disponibilidade de imagens "in-the-wild" e versões pré-processadas (alinhadas e recortadas) para facilitar o uso.

## Use Cases
- **Reconhecimento de Atributos Faciais**: Treinamento de modelos para identificar atributos como idade, gênero, presença de barba, óculos, etc.
- **Reconhecimento Facial**: Desenvolvimento e avaliação de sistemas de identificação de indivíduos.
- **Localização de Pontos de Referência (Landmark Localization)**: Treinamento de modelos para localizar pontos-chave na face.
- **Edição e Síntese Facial**: Geração de novas faces ou modificação de atributos faciais (por exemplo, usando GANs ou VAEs).
- **Pesquisa de Viés e Justiça (Fairness)**: Análise de viés em modelos de IA devido à distribuição de atributos no dataset.

## Integration
O dataset pode ser baixado diretamente dos links fornecidos pela fonte oficial (Google Drive ou Baidu Drive). Para uso em frameworks de aprendizado de máquina, bibliotecas como o **Torchvision** (para PyTorch) e o **TensorFlow Datasets (TFDS)** oferecem APIs para download e carregamento simplificados do CelebA, facilitando a integração em pipelines de treinamento. Por exemplo, no PyTorch, a classe `torchvision.datasets.CelebA` pode ser usada para baixar e carregar o dataset automaticamente. O uso requer a aceitação do acordo de não-uso comercial.

## URL
[https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
