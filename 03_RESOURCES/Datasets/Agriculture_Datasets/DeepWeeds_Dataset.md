# DeepWeeds Dataset

## Description

O DeepWeeds é um dataset de imagens multiclasse, público e anotado, projetado para o reconhecimento robusto de espécies de ervas daninhas em ambientes de pastagem na Austrália. Foi criado para abordar a necessidade de dados de campo realistas para o desenvolvimento de sistemas de controle robótico de ervas daninhas. O dataset é amplamente utilizado em pesquisas de aprendizado profundo (Deep Learning) para classificação de imagens na agricultura de precisão.

## Statistics

**Total de Imagens:** 17.509 imagens coloridas únicas.
**Resolução da Imagem:** 256x256 pixels.
**Divisão Padrão:**
*   **Treinamento:** 60% (aproximadamente 10.505 imagens)
*   **Validação:** 20% (aproximadamente 3.502 imagens)
*   **Teste:** 20% (aproximadamente 3.502 imagens)
**Classes:** 9 classes (8 espécies de ervas daninhas e 1 classe de fundo/outras plantas).
**Balanceamento:** O dataset foi coletado com o objetivo de ter pelo menos 1.000 imagens por espécie e uma divisão equilibrada entre amostras positivas (ervas daninhas) e negativas (fundo/outras plantas) por local.

## Features

**Tipo de Dado:** Imagens coloridas (RGB) de 256x256 pixels.
**Classes:** 9 classes no total (8 espécies de ervas daninhas e 1 classe de "Outras" plantas/fundo).
**Espécies de Ervas Daninhas:** Maçã-de-cheiro (_Ziziphus mauritiana_), Lantana (_Lantana camara_), Parkinsonia (_Parkinsonia aculeata_), Parthenium (_Parthenium hysterophorus_), Acácia-espinhosa (_Vachellia nilotica_), Cipó-de-seda (_Cryptostegia grandiflora_), Erva-de-siam (_Chromolaena odorata_) e Erva-de-cobra (_Stachytarpheta spp._).
**Coleta:** Imagens coletadas _in situ_ (no local) em oito ambientes de pastagem no norte da Austrália, refletindo condições ambientais variáveis (iluminação, oclusão, fundo dinâmico).
**Resolução:** Aproximadamente 4 pixels por mm.

## Use Cases

*   **Classificação de Espécies de Ervas Daninhas:** Treinamento e avaliação de modelos de Deep Learning (como CNNs, ResNet, Inception) para identificar automaticamente a espécie de erva daninha presente em uma imagem.
*   **Agricultura de Precisão:** Desenvolvimento de sistemas de visão computacional para robôs ou drones agrícolas, permitindo a aplicação localizada e seletiva de herbicidas (controle de ervas daninhas específico por local).
*   **Pesquisa em Visão Computacional:** Estudo de técnicas de _feature engineering_ e arquiteturas de rede neural para lidar com a variabilidade de fundo, oclusão e condições de iluminação em ambientes de campo.
*   **Transfer Learning:** Utilização do dataset para ajustar modelos pré-treinados em grandes datasets (como ImageNet) para a tarefa específica de classificação de ervas daninhas.

## Integration

O dataset DeepWeeds está disponível em plataformas como Kaggle e TensorFlow Datasets (TFDS), facilitando o acesso e a integração.

**Exemplo de Acesso via Kaggle API (Shell):**
```bash
# Instalar a CLI do Kaggle (se necessário)
# pip install kaggle

# Fazer o download do dataset (requer credenciais Kaggle configuradas)
kaggle datasets download -d imsparsh/deepweeds
unzip deepweeds.zip
```

**Exemplo de Carregamento em Python (TensorFlow Datasets):**
```python
import tensorflow_datasets as tfds

# Carregar o dataset
(ds_train, ds_test), ds_info = tfds.load(
    'deep_weeds',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Imprimir informações
print(ds_info)

# Exemplo de iteração
for image, label in ds_train.take(1):
    print(f"Formato da Imagem: {image.shape}")
    print(f"Rótulo: {label.numpy()}")
```

## URL

https://www.nature.com/articles/s41598-018-38343-3