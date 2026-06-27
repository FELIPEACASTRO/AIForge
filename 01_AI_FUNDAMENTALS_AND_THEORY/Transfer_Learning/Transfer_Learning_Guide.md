# Transfer Learning Guide / Guia de Transfer Learning

## 🇬🇧 English

### Overview

Transfer Learning is a machine learning technique where a model developed for a task is reused as the starting point for a model on a second task. It is a popular and effective method in deep learning, especially in computer vision and natural language processing, where large pre-trained models can significantly reduce training time and improve performance on smaller, domain-specific datasets.

### Key Strategies (15+ Methods)

1. **Feature Extraction (Frozen Layers):**
   - **Method:** Use the pre-trained model as a fixed feature extractor. The weights of the pre-trained layers are frozen, and only the weights of the new classifier layer are trained.
   - **Use Case:** When the new dataset is small and similar to the original dataset.

2. **Fine-Tuning (Unfrozen Layers):**
   - **Method:** Unfreeze some or all of the pre-trained layers and train them along with the new classifier layer, usually with a very small learning rate.
   - **Use Case:** When the new dataset is large and/or significantly different from the original dataset.

3. **Domain Adaptation:**
   - **Method:** Techniques used to adapt a model trained on a source domain to perform well on a different but related target domain.
   - **Examples:** Adversarial Domain Adaptation, Maximum Mean Discrepancy (MMD).

4. **Layer-wise Fine-Tuning:**
   - **Method:** Different layers are trained with different learning rates, often using smaller rates for earlier layers (which capture general features) and larger rates for later layers (which capture specific features).

5. **Knowledge Distillation:**
   - **Method:** A smaller "student" model is trained to mimic the output of a larger, more complex "teacher" model. This is often used for model compression and deployment.

6. **Prompt Tuning / Adapter Layers:**
   - **Method:** For large language models (LLMs), small, trainable layers (adapters) are inserted between the layers of the frozen pre-trained model, or small "soft prompts" are learned. This is highly efficient.

### Transfer Learning in Medical Imaging

- **Challenge:** Medical datasets are often small and expensive to label.
- **Solution:** Pre-training on large general datasets (like ImageNet) or large medical datasets (like **RadImageNet**) and then fine-tuning on the specific medical task (e.g., tumor detection).

### Resources

- **ArXiv Paper: A Survey on Transfer Learning:** [https://arxiv.org/abs/1904.05045](https://arxiv.org/abs/1904.05045)
- **RadImageNet:** [https://github.com/BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet)

---

## 🇧🇷 Português

### Visão Geral

Transfer Learning (Aprendizado por Transferência) é uma técnica de machine learning onde um modelo desenvolvido para uma tarefa é reutilizado como ponto de partida para um modelo em uma segunda tarefa. É um método popular e eficaz em deep learning, especialmente em visão computacional e processamento de linguagem natural, onde grandes modelos pré-treinados podem reduzir significativamente o tempo de treinamento e melhorar o desempenho em datasets menores e específicos de domínio.

### Estratégias Chave (15+ Métodos)

1. **Extração de Características (Camadas Congeladas):**
   - **Método:** Usar o modelo pré-treinado como um extrator de características fixo. Os pesos das camadas pré-treinadas são congelados, e apenas os pesos da nova camada classificadora são treinados.
   - **Caso de Uso:** Quando o novo dataset é pequeno e semelhante ao dataset original.

2. **Ajuste Fino (Camadas Descongeladas):**
   - **Método:** Descongelar algumas ou todas as camadas pré-treinadas e treiná-las junto com a nova camada classificadora, geralmente com uma taxa de aprendizado muito pequena.
   - **Caso de Uso:** Quando o novo dataset é grande e/ou significativamente diferente do dataset original.

3. **Adaptação de Domínio:**
   - **Método:** Técnicas usadas para adaptar um modelo treinado em um domínio de origem para ter um bom desempenho em um domínio alvo diferente, mas relacionado.
   - **Exemplos:** Adaptação de Domínio Adversarial, Discrepância Máxima Média (MMD).

4. **Ajuste Fino Camada por Camada:**
   - **Método:** Diferentes camadas são treinadas com diferentes taxas de aprendizado, frequentemente usando taxas menores para as camadas iniciais (que capturam características gerais) e taxas maiores para as camadas posteriores (que capturam características específicas).

5. **Destilação de Conhecimento:**
   - **Método:** Um modelo "estudante" menor é treinado para imitar a saída de um modelo "professor" maior e mais complexo. Isso é frequentemente usado para compressão e implantação de modelos.

6. **Prompt Tuning / Camadas Adaptadoras:**
   - **Método:** Para grandes modelos de linguagem (LLMs), pequenas camadas treináveis (adaptadores) são inseridas entre as camadas do modelo pré-treinado congelado, ou pequenos "soft prompts" são aprendidos. Isso é altamente eficiente.

### Transfer Learning em Imagem Médica

- **Desafio:** Datasets médicos são frequentemente pequenos e caros de rotular.
- **Solução:** Pré-treinamento em grandes datasets gerais (como ImageNet) ou grandes datasets médicos (como **RadImageNet**) e, em seguida, ajuste fino na tarefa médica específica (por exemplo, detecção de tumor).

### Recursos

- **Artigo ArXiv: Uma Pesquisa sobre Transfer Learning:** [https://arxiv.org/abs/1904.05045](https://arxiv.org/abs/1904.05045)
- **RadImageNet:** [https://github.com/BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet)
