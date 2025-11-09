# ShapeNet (3D Models)

## Description
ShapeNet é um esforço contínuo para estabelecer um dataset de grande escala de modelos 3D ricamente anotados. É uma colaboração entre pesquisadores de Princeton, Stanford e TTIC, organizado de acordo com a hierarquia WordNet. O dataset é fundamental para pesquisa em computação gráfica, visão computacional e robótica. Possui dois subconjuntos principais: ShapeNetCore e ShapeNetSem. O ShapeNet original indexou mais de 3.000.000 de modelos.

## Statistics
*   **ShapeNet (Total):** Mais de 3.000.000 de modelos indexados (em 2015).
*   **ShapeNetCore:** Aproximadamente **51.300** modelos 3D únicos, cobrindo **55** categorias de objetos.
*   **ShapeNetSem:** Aproximadamente **12.000** modelos, abrangendo **270** categorias.
*   **Versão Principal:** O relatório técnico principal é de 2015, mas o dataset continua sendo a base para novas pesquisas e derivações (como o OpenShape, 2023).

## Features
O dataset é composto por modelos 3D em formato OBJ+MTL. É organizado hierarquicamente usando a taxonomia WordNet.
*   **ShapeNetCore:** Subconjunto com modelos 3D limpos e únicos, com anotações de categoria e alinhamento verificadas manualmente. Cobre 55 categorias comuns de objetos.
*   **ShapeNetSem:** Subconjunto menor, mais densamente anotado, com 270 categorias. Inclui anotações de dimensões do mundo real, composição material, volume e peso.
*   **ShapeNetPart:** Uma derivação popular do ShapeNetCore com anotações de segmentação de partes.

## Use Cases
*   Reconhecimento e classificação de objetos 3D.
*   Segmentação de partes de objetos 3D (com ShapeNetPart).
*   Síntese e geração de formas 3D.
*   Estimativa de pose de objetos 3D.
*   Pesquisa em robótica para percepção e manipulação de objetos.
*   Treinamento de modelos de aprendizado profundo para representação de formas volumétricas (3D ShapeNets).

## Integration
O acesso aos dados do modelo (OBJ+MTL) e aos metadados de anotação processados é fornecido para fins de pesquisa e/ou educacionais não comerciais. É necessário **registrar** uma conta no site oficial. Após a verificação e aprovação por um administrador do site, o download é liberado. O ShapeNet Viewer é fornecido para visualização e renderização dos modelos. O artigo principal a ser citado é: "ShapeNet: An Information-Rich 3D Model Repository" (2015).

## URL
[https://shapenet.org/](https://shapenet.org/)
