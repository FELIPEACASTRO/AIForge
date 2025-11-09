# Prefix Tuning

## Description
O **Prefix Tuning** é uma técnica de *Parameter-Efficient Fine-Tuning* (PEFT) que adapta grandes modelos de linguagem (LLMs) para tarefas específicas de Geração de Linguagem Natural (NLG) sem a necessidade de ajustar todos os parâmetros do modelo. Em vez disso, ele mantém os parâmetros do modelo pré-treinado congelados e otimiza um pequeno vetor contínuo e específico da tarefa, chamado de **prefixo**. Este prefixo é inserido em todas as camadas do transformador, atuando como "tokens virtuais" que guiam o modelo para a saída desejada. Ao aprender apenas cerca de 0,1% dos parâmetros, o Prefix Tuning alcança um desempenho comparável ao *fine-tuning* completo em cenários de dados abundantes e o supera em cenários de poucos dados, sendo significativamente mais eficiente em termos de custo computacional e armazenamento. Sua principal inovação é permitir que o modelo atenda a este prefixo como se fossem *tokens* de entrada reais, influenciando o processo de geração em todos os blocos do transformador.

## Examples
```
**Exemplos de Prompts (Conceituais, pois o prefixo é um vetor contínuo):**

1.  **Sumarização de Notícias (BART):**
    *   **Prefix Tuning Treinado:** `[Prefix_Sumarização_Notícias]`
    *   **Prompt de Entrada:** `[Prefix_Sumarização_Notícias] Artigo: O governo anunciou hoje um novo plano de infraestrutura focado em energia renovável. Especialistas preveem um impacto positivo no PIB. Resumo:`
    *   **Saída Esperada:** `O novo plano de infraestrutura do governo prioriza energia renovável, com expectativas de impulsionar o PIB.`

2.  **Geração de Tabela para Texto (GPT-2):**
    *   **Prefix Tuning Treinado:** `[Prefix_Tabela_para_Texto]`
    *   **Prompt de Entrada:** `[Prefix_Tabela_para_Texto] Tabela: | Ator | Filme | Ano | | Tom Hanks | Forrest Gump | 1994 | | Meryl Streep | O Diabo Veste Prada | 2006 | Sentença:`
    *   **Saída Esperada:** `Tom Hanks estrelou Forrest Gump em 1994, e Meryl Streep participou de O Diabo Veste Prada em 2006.`

3.  **Tradução de Idiomas (Inglês para Português):**
    *   **Prefix Tuning Treinado:** `[Prefix_Tradução_EN_PT]`
    *   **Prompt de Entrada:** `[Prefix_Tradução_EN_PT] Traduza: "The quick brown fox jumps over the lazy dog." Tradução:`
    *   **Saída Esperada:** `A rápida raposa marrom salta sobre o cão preguiçoso.`

4.  **Classificação de Sentimento (Análise de Avaliações):**
    *   **Prefix Tuning Treinado:** `[Prefix_Sentimento_Positivo]`
    *   **Prompt de Entrada:** `[Prefix_Sentimento_Positivo] Avaliação: "Este produto superou todas as minhas expectativas, a qualidade é fantástica." Sentimento:`
    *   **Saída Esperada:** `Positivo`

5.  **Geração de Código (Python):**
    *   **Prefix Tuning Treinado:** `[Prefix_Geração_Python_Flask]`
    *   **Prompt de Entrada:** `[Prefix_Geração_Python_Flask] Crie um endpoint Flask para retornar a data e hora atuais. Código:`
    *   **Saída Esperada:** `from flask import Flask\nfrom datetime import datetime\napp = Flask(__name__)\n@app.route('/time')\ndef get_time():\n    return {'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

6.  **Resposta a Perguntas (Domínio Médico):**
    *   **Prefix Tuning Treinado:** `[Prefix_QA_Médico]`
    *   **Prompt de Entrada:** `[Prefix_QA_Médico] Pergunta: Qual é o principal sintoma da apendicite aguda? Resposta:`
    *   **Saída Esperada:** `Dor abdominal intensa, geralmente começando ao redor do umbigo e migrando para o quadrante inferior direito.`

7.  **Geração de Diálogo (Estilo Shakespeareano):**
    *   **Prefix Tuning Treinado:** `[Prefix_Diálogo_Shakespeare]`
    *   **Prompt de Entrada:** `[Prefix_Diálogo_Shakespeare] Personagem A: Diga-me, meu senhor, o que vos aflige a alma? Personagem B:`
    *   **Saída Esperada:** `Ah, é a própria melancolia que me rouba o sono e me pesa o coração, meu caro amigo.`
```

## Best Practices
**Melhores Práticas (Best Practices):**
1.  **Congelamento de Parâmetros:** Mantenha os parâmetros do modelo pré-treinado congelados para garantir a eficiência e evitar o *catastrophic forgetting*.
2.  **Comprimento do Prefixo:** O comprimento do prefixo (número de *virtual tokens*) é um hiperparâmetro crucial. Comece com comprimentos curtos e aumente gradualmente, validando o desempenho.
3.  **Projeção do Prefixo:** Utilize a projeção do prefixo (uma MLP de duas camadas) para mitigar o risco de *overfitting* e melhorar a capacidade de generalização, especialmente em tarefas mais complexas.
4.  **Configuração de Baixos Recursos:** Priorize o Prefix Tuning em cenários com dados limitados ou recursos computacionais restritos, onde o *fine-tuning* completo seria inviável.
5.  **Combinação com LoRA:** Considere a combinação com outras técnicas PEFT, como o LoRA, para otimizar ainda mais o número de parâmetros treináveis e o controle específico da tarefa.

## Use Cases
**Casos de Uso (Use Cases):**
1.  **Geração de Linguagem Natural (NLG):** Ideal para tarefas como sumarização, tradução, resposta a perguntas e geração de diálogos, onde o modelo precisa ser guiado para um estilo ou formato de saída específico.
2.  **Adaptação de Domínio:** Ajustar um LLM pré-treinado para funcionar de forma eficaz em um domínio específico (e.g., jurídico, médico, financeiro) com um conjunto limitado de dados de treinamento.
3.  **Cenários de Baixos Recursos:** Aplicações em que o custo computacional ou o armazenamento de múltiplos modelos *fine-tuned* é proibitivo.
4.  **Aprendizado Multitarefa:** Treinar um único prefixo para lidar com múltiplas tarefas relacionadas, aproveitando a eficiência de parâmetros para alternar rapidamente entre elas.
5.  **Personalização de Modelos:** Criar versões personalizadas de um LLM base para diferentes usuários ou clientes, cada um com seu próprio prefixo leve.

## Pitfalls
**Armadilhas Comuns (Pitfalls):**
1.  **Overfitting do Prefixo:** Se o prefixo for muito longo ou o conjunto de dados de treinamento for muito pequeno, o prefixo pode se ajustar demais aos dados de treinamento, perdendo a capacidade de generalização.
2.  **Seleção Inadequada de Hiperparâmetros:** A escolha incorreta do comprimento do prefixo ou da taxa de aprendizado pode levar a um desempenho subótimo.
3.  **Complexidade da Tarefa:** Para tarefas que exigem uma modificação profunda do conhecimento interno do modelo (em vez de apenas uma mudança no estilo ou formato de saída), o Prefix Tuning pode não ser tão eficaz quanto o *fine-tuning* completo.
4.  **Falta de Interpretabilidade:** O prefixo é um vetor contínuo, o que significa que não é legível por humanos. Isso torna o processo de depuração e otimização mais desafiador do que com *prompts* de texto discretos.
5.  **Problemas de Memória (Atenção):** Embora seja eficiente em termos de parâmetros, o Prefix Tuning pode aumentar o custo de memória de ativação durante a inferência, pois o prefixo precisa ser concatenado e processado em cada camada do transformador.

## URL
[https://arxiv.org/abs/2101.00190](https://arxiv.org/abs/2101.00190)
