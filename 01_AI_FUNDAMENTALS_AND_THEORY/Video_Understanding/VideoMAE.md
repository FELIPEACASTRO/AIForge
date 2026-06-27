# VideoMAE

## Description

VideoMAE (Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training) é um framework de aprendizado auto-supervisionado que estende o conceito de Masked Autoencoders (MAE) para o domínio de vídeo. Sua proposta de valor única é demonstrar que MAEs são aprendizes eficientes em dados para pré-treinamento de vídeo, utilizando uma taxa de mascaramento extremamente alta (90%-95%) e uma estratégia de mascaramento de tubo (tube masking) para forçar o modelo a aprender a coerência espaço-temporal, superando a redundância inerente aos vídeos.

## Statistics

O modelo base (VideoMAE-Base) atinge cerca de **80.9%** de precisão Top-1 e **94.7%** de precisão Top-5 no conjunto de testes Kinetics-400 (após fine-tuning). Versões mais recentes (VideoMAE V2 Huge) alcançam até **86.6%** de precisão Top-1 e **97.1%** de precisão Top-5 no Kinetics-400.

## Features

Estratégia de Mascaramento de Tubo (Tube Masking); Alta Taxa de Mascaramento (90%-95%); Arquitetura de Transformer (ViT) com codificador e decodificador leve; Aprendizado auto-supervisionado eficiente em dados.

## Use Cases

Classificação de Ação em Vídeo (Kinetics-400); Detecção de Ação; Reconhecimento de Atividades Humanas (HAR) em robótica assistiva e vigilância; Backbone para diversas tarefas de compreensão de vídeo.

## Integration

O modelo é facilmente acessível e utilizável através da biblioteca Hugging Face Transformers.\n\n```python\nfrom transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification\nimport torch\n\n# Carregar o extrator de características e o modelo pré-treinado\nfeature_extractor = VideoMAEFeatureExtractor.from_pretrained(\"MCG-NJU/videomae-base-finetuned-kinetics\")\nmodel = VideoMAEForVideoClassification.from_pretrained(\"MCG-NJU/videomae-base-finetuned-kinetics\")\n\n# Exemplo de uso (substituir 'video_data' pelos frames do seu vídeo)\n# inputs = feature_extractor(video_data, return_tensors=\"pt\")\n# with torch.no_grad():\n#     outputs = model(**inputs)\n# logits = outputs.logits\n# predicted_class_idx = logits.argmax(-1).item()\n# print(f\"Classe Prevista: {model.config.id2label[predicted_class_idx]}\")\n```

## URL

https://github.com/MCG-NJU/VideoMAE