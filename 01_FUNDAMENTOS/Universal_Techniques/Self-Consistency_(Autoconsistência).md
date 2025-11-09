# Self-Consistency (Autoconsistência)

## Description
A Autoconsistência (Self-Consistency - SC) é uma técnica avançada de Engenharia de Prompt que aprimora a capacidade de raciocínio de Grandes Modelos de Linguagem (LLMs), especialmente em tarefas complexas de raciocínio aritmético e de senso comum. Proposta como uma estratégia de decodificação que substitui a decodificação gulosa (greedy decoding) ingênua, a SC funciona gerando múltiplas e diversas 'cadeias de pensamento' (Chain-of-Thought - CoT) para uma única pergunta. Em vez de aceitar a primeira resposta gerada, a técnica agrega os resultados de todos os caminhos de raciocínio amostrados e seleciona a resposta final por meio de um voto majoritário (ou outra métrica de consistência). Isso capitaliza a ideia de que, embora um problema complexo possa ter vários caminhos de raciocínio válidos, todos eles devem levar a uma única resposta correta, aumentando significativamente a precisão e a robustez da solução. A variante mais avançada, Autoconsistência Universal (Universal Self-Consistency - USC), utiliza o próprio LLM para atuar como um juiz, selecionando a resposta mais consistente entre os candidatos.

## Examples
```
## Exemplos de Prompts (Self-Consistency)

**Instrução Geral:** Para aplicar a Autoconsistência, o prompt deve ser enviado ao LLM *N* vezes (onde N é o número de amostras desejado, tipicamente 5 a 10), e a resposta final é determinada pelo voto majoritário entre as *N* respostas.

### Exemplo 1: Raciocínio Aritmético (GSM8K)

**Prompt (a ser repetido N vezes):**

```
Q: O João tem 15 maçãs. Ele come 3 e dá metade das restantes para a Maria. Quantas maçãs a Maria recebeu?

Vamos pensar passo a passo:

[Espaço para o LLM gerar o raciocínio e a resposta final. O prompt deve ser projetado para forçar o raciocínio CoT.]
```

**Processo:**
1. Enviar o prompt 10 vezes.
2. Coletar as 10 respostas finais (ex: 7, 6, 6, 6, 7, 6, 6, 7, 6, 6).
3. A resposta final é 6 (voto majoritário).

### Exemplo 2: Raciocínio de Senso Comum (StrategyQA)

**Prompt (a ser repetido N vezes):**

```
Q: É possível que um adulto durma mais de 12 horas por dia regularmente sem ter uma condição médica?

Vamos pensar passo a passo:

[Espaço para o LLM gerar o raciocínio e a resposta final (Sim/Não).] 
```

**Processo:**
1. Enviar o prompt 5 vezes.
2. Coletar as 5 respostas finais (ex: Não, Sim, Não, Não, Não).
3. A resposta final é Não.

### Exemplo 3: Universal Self-Consistency (USC) - Etapa de Julgamento

**Prompt de Julgamento (enviado a um LLM 'Juiz' após gerar N respostas):**

```
Você é um juiz de consistência. Analise as seguintes N respostas para a pergunta 'Qual é a capital do Butão?' e selecione a mais consistente e correta. Justifique sua escolha.

Respostas Candidatas:
1. Thimphu, pois é o centro político e econômico.
2. Paro, devido ao seu aeroporto internacional.
3. Thimphu, a maior cidade e capital oficial.

Resposta Final e Justificativa:
```

### Exemplo 4: Tarefa de Classificação Complexa

**Prompt (a ser repetido N vezes):**

```
Q: Classifique o seguinte texto como 'Notícia', 'Opinião' ou 'Publicidade', e justifique sua escolha:

[TEXTO: 'O novo smartphone X é o mais rápido do mercado, com uma câmera que redefine a fotografia móvel. Disponível agora por um preço imbatível.']

Vamos pensar passo a passo:

[Espaço para o LLM gerar o raciocínio e a classificação final.]
```

### Exemplo 5: Resolução de Quebra-Cabeças Lógicos

**Prompt (a ser repetido N vezes):**

```
Q: Se o código para 'SOL' é 191512 e o código para 'LUA' é 122101, qual é o código para 'MAR'?

Vamos pensar passo a passo:

[Espaço para o LLM gerar o raciocínio e o código final.]
```
```

## Best Practices
1. **Aumente a Diversidade:** Use um parâmetro de temperatura (temperature) mais alto (ex: 0.7 a 1.0) ao gerar os caminhos de raciocínio para garantir que as amostras sejam diversas.
2. **Amostragem Suficiente:** O número de amostras (N) deve ser grande o suficiente (tipicamente N=5 a N=10) para que o voto majoritário seja estatisticamente significativo.
3. **CoT é Fundamental:** A Autoconsistência deve ser combinada com o Chain-of-Thought (CoT) para forçar o LLM a gerar os passos de raciocínio que levam à resposta.
4. **Foco na Resposta Final:** A votação deve ser aplicada apenas à resposta final extraída de cada caminho de raciocínio, e não ao raciocínio em si.
5. **Use USC para Maior Precisão:** Para tarefas críticas, utilize a Autoconsistência Universal (USC), onde um segundo LLM é usado para julgar e selecionar a melhor resposta, em vez de um simples voto majoritário.

## Use Cases
1. **Raciocínio Aritmético e Matemático:** Resolução de problemas de palavras complexos (como o benchmark GSM8K).
2. **Raciocínio de Senso Comum:** Respostas a perguntas que exigem inferência e conhecimento do mundo (como o benchmark StrategyQA).
3. **Verificação de Fatos e Dados:** Aumentar a confiança em respostas factuais, especialmente em domínios com alta ambiguidade.
4. **Classificação e Análise de Sentimento:** Melhorar a precisão em tarefas de classificação complexas, onde o contexto pode levar a diferentes interpretações.
5. **Aplicações de Agentes de IA:** Como um mecanismo de verificação de robustez para a tomada de decisões em agentes autônomos.

## Pitfalls
1. **Custo Computacional Elevado:** Requer N vezes mais chamadas à API do LLM, aumentando significativamente o custo e o tempo de latência.
2. **Dependência do CoT:** A eficácia da SC está intrinsecamente ligada à qualidade dos caminhos de raciocínio gerados pelo CoT. Se o CoT falhar, a SC também falhará.
3. **Voto Majoritário Falho:** Em casos de respostas muito dispersas, o voto majoritário pode não ser claro ou pode convergir para uma resposta incorreta se a maioria dos caminhos de raciocínio tiver o mesmo erro sutil.
4. **Dificuldade de Implementação:** Requer uma etapa de pós-processamento para extrair a resposta final de cada saída e realizar a votação, o que é mais complexo do que a decodificação gulosa simples.
5. **Não Adequado para Tarefas Criativas:** Não é recomendado para tarefas que valorizam a diversidade de saída (ex: geração de poesia, brainstorming), pois seu objetivo é convergir para uma única resposta correta.

## URL
[https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
