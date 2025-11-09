# Modelo de Conjunto com Transfer Learning para Detecção Precoce da Doença de Alzheimer (DA)

## Description

Um robusto *framework* de aprendizado profundo que emprega *transfer learning* e ajuste de hiperparâmetros das arquiteturas InceptionResnetV2, InceptionV3 e Xception. Utiliza um mecanismo de voto de conjunto (*ensemble voting*) para combinar as previsões e otimizar a acurácia e robustez na classificação de quatro estágios da Doença de Alzheimer (DA): Não-Demente, Demente Muito Leve, Demente Leve e Demente Moderado. O modelo de conjunto superou todos os modelos de linha de base individuais.

## Statistics

- **Acurácia Geral:** 98.96%
- **Precisão (Mildly Demented):** 100%
- **Precisão (Moderately Demented):** 100%
- **Recall (Mildly Demented):** 100%
- **Recall (Moderately Demented):** 100%
- **Métricas de Desempenho:** O modelo de conjunto alcançou um desempenho superior em todas as métricas (Acurácia, Precisão, Recall, F1-Score) em comparação com os modelos base individuais (InceptionResNetV2, InceptionV3, Xception).
- **Misclassificações:** Apenas 9 misclassificações de amostras "Demente Muito Leve" como "Não-Demente", o menor erro entre todas as arquiteturas avaliadas.
- **Publicação:** Scientific Reports, Volume 15, Artigo número: 34634 (2025).

## Features

- **Arquitetura de Conjunto Ponderado:** Combina as previsões de três modelos CNN pré-treinados (InceptionResNetV2, InceptionV3, Xception) usando um mecanismo de voto ponderado (*weighted voting*).
- **Transfer Learning:** Utiliza modelos pré-treinados no ImageNet para extração de características, com ajuste fino (*fine-tuning*) das camadas finais para a classificação específica da DA.
- **Classificação Multi-Classe:** Classifica imagens de ressonância magnética (MRI) em quatro categorias de progressão da DA.
- **Pré-processamento e Aumento de Dados:** Inclui redimensionamento, conversão para escala de cinza e técnicas de aumento de dados (inversão horizontal, zoom, cisalhamento) para mitigar o desequilíbrio de dados e melhorar a generalização.

## Use Cases

- **Diagnóstico Precoce da Doença de Alzheimer (DA):** Classificação de estágios da DA (Não-Demente, Demente Muito Leve, Demente Leve, Demente Moderado) a partir de imagens de ressonância magnética (MRI).
- **Apoio à Decisão Clínica:** Fornecer uma segunda opinião automatizada e robusta para radiologistas e neurologistas, auxiliando na intervenção e manejo precoces da doença.
- **Pesquisa em Neuroimagem:** Servir como uma arquitetura de linha de base de alto desempenho para futuras pesquisas que buscam incorporar dados multimodais (clínicos, genéticos) para um diagnóstico ainda mais preciso.

## Integration

O código-fonte do estudo está disponível em um repositório público do GitHub, o que facilita a integração e reprodução do modelo. O *framework* é implementado em um ambiente Google Colab, sugerindo o uso de bibliotecas populares de aprendizado profundo como TensorFlow/Keras.

**Recursos de Código:**
- **Repositório GitHub:** `https://github.com/muhammadmo/Alzheimer_Classification_MRI`
- **Conjunto de Dados:** O estudo utilizou o conjunto de dados ADNI, disponível no Kaggle: `https://www.kaggle.com/datasets/praneshkumarm/multidiseasedataset`

## URL

https://www.nature.com/articles/s41598-025-22025-y