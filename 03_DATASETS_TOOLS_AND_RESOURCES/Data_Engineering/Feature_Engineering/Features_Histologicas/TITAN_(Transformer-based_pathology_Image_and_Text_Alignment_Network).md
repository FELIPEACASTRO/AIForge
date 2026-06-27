# TITAN (Transformer-based pathology Image and Text Alignment Network)

## Description

Um modelo de fundação multimodal de lâmina inteira (whole-slide) para patologia, pré-treinado usando 335.645 imagens de lâminas inteiras (WSIs) através de aprendizado autossupervisionado visual e alinhamento visão-linguagem com relatórios de patologia correspondentes e 423.122 legendas sintéticas. Ele extrai representações de lâmina de propósito geral e gera relatórios de patologia sem a necessidade de fine-tuning ou rótulos clínicos.

## Statistics

Pré-treinado em 335.645 WSIs em 20 tipos de órgãos. Utiliza 423.122 legendas sintéticas de ROI (Região de Interesse) de grão fino e 183 mil relatórios de patologia para fine-tuning visão-linguagem. Codifica milhões de ROIs de alta resolução (8.192 × 8.192 pixels com ampliação de 20×).

## Features

Alinhamento Multimodal (Imagem e Texto). Aprendizado de representação de lâmina inteira. Classificação zero-shot. Recuperação cross-modal (lâminas histológicas e relatórios clínicos). Geração de relatórios de patologia. Supera modelos de fundação de ROI e de lâmina em várias tarefas.

## Use Cases

Aprendizado de representação de lâmina de propósito geral, subtipagem de câncer, previsão de biomarcadores, prognóstico de resultados, recuperação de lâminas, recuperação de câncer raro, classificação zero-shot guiada por linguagem.

## Integration

O modelo é um modelo de fundação, sugerindo que suas features podem ser extraídas e usadas em tarefas subsequentes. O artigo menciona que ele pode ser aplicado 'pronto para uso' (off-the-shelf) para previsão de desfechos clínicos. Detalhes adicionais sobre a disponibilidade do código devem ser consultados na seção 'Code availability' do artigo.

## URL

https://www.nature.com/articles/s41591-025-03982-3