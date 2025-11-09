# TextVQA

## Description
O **TextVQA** é um dataset de Question Answering Visual (VQA) que exige que os modelos leiam e raciocinem sobre o texto presente nas imagens para responder a perguntas sobre elas. Foi criado para abordar a limitação de datasets VQA anteriores, que possuíam uma pequena proporção de perguntas que exigiam a leitura de texto. O dataset é fundamental para o desenvolvimento de modelos que incorporam a modalidade de texto de cena (scene text) para responder a perguntas. O objetivo é estimular o progresso em modelos VQA que possam "ler" o texto embutido em imagens.

## Statistics
**Versão Principal:** v0.5.1 (e v0.5, que é idêntica exceto pelos tokens de OCR).
**Total de Imagens:** 28.408 imagens (do OpenImages).
**Total de Perguntas:** 45.336 perguntas.
**Total de Respostas:** 453.360 respostas de verdade fundamental (10 por pergunta).
**Divisão do Dataset (v0.5.1):**
*   **Treino:** 34.602 perguntas, 21.953 imagens (6.6GB).
*   **Validação:** 5.000 perguntas, 3.166 imagens.
*   **Teste:** 5.734 perguntas, 3.289 imagens (926MB).
**Tamanho:** O conjunto de dados de perguntas/respostas é relativamente pequeno (cerca de 132MB no total), mas o conjunto de imagens é de aproximadamente 7.5GB.

## Features
**Natureza Multimodal:** Combina visão computacional e Processamento de Linguagem Natural (NLP), com foco em texto de cena. **Requisito de Leitura:** As perguntas são formuladas de modo que a resposta só possa ser determinada pela leitura e compreensão do texto visível na imagem. **OCR Integrado:** Fornece tokens de OCR (Optical Character Recognition) extraídos pelo sistema Rosetta, além das anotações de perguntas e respostas. **Base de Imagens:** As imagens são provenientes do dataset OpenImages. **Avaliação:** A avaliação é realizada através do servidor EvalAI, utilizando a métrica de acurácia.

## Use Cases
**Question Answering Visual (VQA) com Texto de Cena:** Treinamento e avaliação de modelos de VQA que precisam extrair e compreender informações textuais em imagens. **Reconhecimento Óptico de Caracteres (OCR) em Contexto:** Desenvolvimento de sistemas de OCR mais robustos que operam em cenários do mundo real e integram o contexto visual. **Modelos de Linguagem e Visão (VLMs):** Benchmarking de modelos multimodais que integram texto, imagem e texto de cena para raciocínio complexo. **Assistência a Pessoas com Deficiência Visual:** O estudo original aponta que uma classe dominante de perguntas feitas por usuários com deficiência visual envolve a leitura de texto em imagens do seu entorno.

## Integration
O dataset TextVQA está disponível para download no site oficial e no Hugging Face. A versão recomendada é a **v0.5.1**. Os dados são divididos em arquivos JSON para perguntas/respostas e arquivos ZIP para as imagens e tokens de OCR.
1.  **Download:** Os arquivos de perguntas/respostas e os tokens de OCR (Rosetta OCR tokens \[v0.2\]) estão disponíveis em formato JSON. As imagens (provenientes do OpenImages) são fornecidas em arquivos ZIP separados para os conjuntos de treino e teste.
2.  **Estrutura:** Os arquivos JSON contêm a `question_id`, a `question`, o `image_id` (do OpenImages), e até 10 `answers` (respostas de verdade fundamental).
3.  **Uso:** Os pesquisadores são encorajados a usar seus próprios sistemas de OCR, embora os tokens de OCR fornecidos sejam úteis para a linha de base. A submissão de resultados para avaliação é feita através do servidor EvalAI.
4.  **Licença:** O dataset está disponível sob a licença **CC BY 4.0**.

## URL
[https://textvqa.org/](https://textvqa.org/)
