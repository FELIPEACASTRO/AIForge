# EffiDec3D

## Description

EffiDec3D é um decodificador 3D otimizado para a **segmentação de imagens médicas 3D de alto desempenho e eficiente**. Foi proposto para resolver o alto custo computacional (alto número de #FLOPs e #Params) de redes 3D profundas existentes, como SwinUNETR e 3D UX-Net, que limitam seu uso em ambientes de tempo real e com recursos limitados. O modelo emprega uma estratégia de **redução de canais** em todos os estágios do decodificador e **remove camadas de alta resolução** quando sua contribuição para a qualidade da segmentação é mínima. Esta abordagem estabelece um novo padrão para a segmentação eficiente de imagens médicas 3D, mantendo um desempenho comparável aos modelos originais, mas com uma fração dos recursos computacionais.

## Statistics

- **Redução de Parâmetros (#Params):** 96,4% de redução em comparação com o decodificador do 3D UX-Net original.
- **Redução de Operações de Ponto Flutuante (#FLOPs):** 93,0% de redução em comparação com o decodificador do 3D UX-Net original.
- **Desempenho:** Mantém um nível de desempenho comparável aos modelos originais (SwinUNETR, 3D UX-Net) em 12 tarefas diferentes de imagens médicas.
- **Publicação:** Apresentado na **CVPR 2025** (Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition).

## Features

- **Decodificador 3D Otimizado:** Focado na eficiência computacional.
- **Estratégia de Redução de Canais:** Define o número mínimo de canais necessários para uma representação precisa dos recursos.
- **Remoção de Camadas de Alta Resolução:** Elimina camadas com contribuição mínima para a qualidade da segmentação.
- **Compatibilidade:** Pode ser integrado com codificadores existentes (e.g., SwinUNETR, 3D UX-Net).
- **Segmentação Volumétrica:** Especializado em dados de imagem médica 3D.

## Use Cases

- **Segmentação de Imagens Médicas 3D:** Aplicação primária em tarefas de segmentação volumétrica, como a identificação de órgãos e anomalias em exames de ressonância magnética (MRI) e tomografia computadorizada (CT).
- **Ambientes com Recursos Limitados:** Ideal para implantação em dispositivos de borda ou em sistemas que exigem processamento em tempo real devido à sua alta eficiência computacional.
- **Pesquisa em Eficiência de DL:** Serve como um *benchmark* para o desenvolvimento de arquiteturas de decodificadores mais eficientes em redes neurais convolucionais 3D.

## Integration

A implementação oficial em PyTorch está disponível no GitHub. O código inclui scripts de treinamento para datasets como BTCV e MSD (Task01-10), indicando que a integração se dá através da utilização da arquitetura EffiDec3D em um pipeline de treinamento PyTorch, substituindo o decodificador original de modelos como SwinUNETR ou 3D UX-Net.
**Link para o Repositório:** [https://github.com/SLDGroup/EffiDec3D](https://github.com/SLDGroup/EffiDec3D)

## URL

https://openaccess.thecvf.com/content/CVPR2025/html/Rahman_EffiDec3D_An_Optimized_Decoder_for_High-Performance_and_Efficient_3D_Medical_CVPR_2025_paper.html