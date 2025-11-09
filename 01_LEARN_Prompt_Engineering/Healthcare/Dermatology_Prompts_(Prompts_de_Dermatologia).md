# Dermatology Prompts (Prompts de Dermatologia)

## Description
**Dermatology Prompts** (Prompts de Dermatologia) referem-se a comandos de engenharia de prompt especificamente elaborados para interagir com Modelos de Linguagem Grande (LLMs) e Modelos Multimodais (LMMs) no contexto da dermatologia. O objetivo é transformar a IA em um assistente clínico, educacional ou de pesquisa, capaz de auxiliar em tarefas como diagnóstico diferencial, planejamento terapêutico, interpretação de exames, educação médica e otimização de rotinas administrativas. A eficácia desses prompts reside na sua capacidade de fornecer contexto clínico detalhado, definir o papel da IA como um especialista e exigir um formato de saída estruturado e referenciado. A área ganhou destaque a partir de 2023, com o avanço de modelos como o GPT-4, que demonstraram alta precisão na geração de vinhetas clínicas e suporte à decisão, desde que o prompt seja construído com rigor e atenção à ética e à necessidade de validação humana. O uso de prompts em dermatologia exige cautela, sendo a IA uma ferramenta de suporte e não um substituto para o julgamento clínico do médico.

## Examples
```
**1. Diagnóstico Diferencial Estruturado (Role + Task + Format)**
```
Aja como um dermatologista experiente. Analise o seguinte caso: Paciente masculino, 45 anos, apresenta pápulas eritematosas e pruriginosas no tronco e membros, com distribuição simétrica, há 3 semanas. Histórico de estresse recente. Sem febre ou sintomas sistêmicos.
Tarefa: Liste 5 diagnósticos diferenciais mais prováveis, do mais comum ao mais raro. Para cada um, crie uma tabela com os seguintes campos: 'Diagnóstico', 'Sinais Chave de Diferenciação', 'Exame Complementar Sugerido'.
```

**2. Otimização de Plano Terapêutico (Refinamento)**
```
Aja como um farmacologista especializado em dermatologia. O paciente foi diagnosticado com Psoríase em Placas (PASI 15). O plano inicial é Metotrexato 15mg/semana.
Tarefa: Avalie a segurança e eficácia deste plano. Sugira 3 alternativas de tratamento de segunda linha (biológicos ou orais) e crie um prompt de acompanhamento para o paciente, explicando os efeitos colaterais do Metotrexato em linguagem leiga.
```

**3. Análise de Imagem (Prompt Multimodal)**
```
Aja como um dermatoscopista. A imagem anexada mostra uma lesão pigmentada no dorso.
Tarefa: Descreva a lesão usando a regra ABCDE (Assimetria, Bordas, Cor, Diâmetro, Evolução). Com base na descrição, qual é a hipótese diagnóstica mais provável (Melanoma, Nevo Displásico ou Ceratose Seborreica)? Justifique sua resposta com base nos critérios dermatoscópicos e sugira o próximo passo (biópsia excisional ou acompanhamento).
```

**4. Criação de Vinheta Clínica para Educação Médica**
```
Aja como um examinador do USMLE Step 2 CK.
Tarefa: Crie uma vinheta clínica de 150 palavras sobre um caso de Dermatite Atópica em um paciente pediátrico, focando em fatores desencadeantes e manejo inicial. A vinheta deve ser seguida por uma pergunta de múltipla escolha sobre o tratamento de primeira linha.
```

**5. Pesquisa e Síntese de Artigos Científicos**
```
Aja como um pesquisador sênior.
Tarefa: Pesquise no PubMed os 3 artigos mais recentes (2024-2025) sobre o uso de Inteligência Artificial para detecção precoce de Câncer de Pele Não-Melanoma. Resuma os principais achados, a metodologia utilizada (tipo de IA) e a taxa de acurácia (AUC). Apresente o resultado em formato de tabela.
```

**6. Protocolo de Comunicação com o Paciente**
```
Aja como um especialista em comunicação médica.
Tarefa: Elabore um texto conciso e empático para ser entregue a um paciente recém-diagnosticado com Vitiligo, explicando a condição, as opções de tratamento (fototerapia, tópicos) e a importância do suporte psicológico. O tom deve ser informativo e encorajador.
```
```

## Best Practices
**1. Defina o Papel (Role) e o Contexto:** Sempre comece o prompt definindo o papel da IA (ex: "Aja como um dermatologista com 20 anos de experiência e foco em oncologia cutânea"). Isso direciona o tom e a base de conhecimento. **2. Estrutura Padrão (RTF):** Utilize o framework **Role, Task, Format** (Papel, Tarefa, Formato). Exija o formato de saída (tabela, lista, resumo) para estruturar a resposta. **3. Anonimização e Ética:** **NUNCA** insira dados de pacientes que possam identificá-los. Use apenas dados clínicos anonimizados ou cenários hipotéticos. **4. Exija Referências:** Peça à IA para citar as fontes (ex: "Baseie sua resposta nas diretrizes da Sociedade Brasileira de Dermatologia e cite os artigos do PubMed"). **5. Profundidade e Detalhe:** Para diagnósticos complexos, solicite uma "investigação profunda" (deep research) para obter relatórios mais longos e embasados. **6. Use Modelos Multimodais:** Para análise de lesões, utilize modelos que aceitam imagens (GPT-4V, Gemini Pro Vision), descrevendo a imagem com o máximo de detalhes clínicos possível no prompt.

## Use Cases
nan

## Pitfalls
**1. Alucinações e Falsas Referências:** A IA pode "alucinar" (gerar informações falsas) ou citar artigos e diretrizes inexistentes. **Contramedida:** Sempre exija a citação de fontes e verifique-as manualmente, especialmente em decisões críticas. **2. Violação de Sigilo:** Inserir dados de pacientes que possam identificá-los (nome, CPF, endereço) em IAs públicas. **Contramedida:** Use apenas dados 100% anonimizados ou cenários hipotéticos. **3. Falta de Contexto Clínico:** Prompts muito curtos ou genéricos (ex: "O que é acne?") resultam em respostas superficiais e inúteis para a prática clínica. **Contramedida:** Forneça o máximo de detalhes clínicos, demográficos e de histórico do paciente. **4. Confiança Excessiva (Automation Bias):** Aceitar a sugestão da IA sem o devido raciocínio clínico e validação humana. **Contramedida:** A IA é um assistente; a responsabilidade final pelo diagnóstico e tratamento é sempre do médico. **5. Limitação Multimodal:** Modelos de IA podem ter dificuldade em interpretar nuances em imagens de baixa qualidade ou com iluminação inadequada. **Contramedida:** Complemente a imagem com uma descrição clínica detalhada no prompt.

## URL
[https://www.nature.com/articles/s41746-025-01650-x](https://www.nature.com/articles/s41746-025-01650-x)
