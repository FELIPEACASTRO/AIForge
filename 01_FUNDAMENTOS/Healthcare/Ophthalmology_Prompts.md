# Ophthalmology Prompts

## Description
A Engenharia de Prompt em Oftalmologia refere-se à arte e ciência de formular entradas (prompts) para Modelos de Linguagem Grande (LLMs) com o objetivo de otimizar a precisão, relevância e utilidade das respostas no contexto da saúde ocular. Esta técnica é crucial para alavancar o potencial da IA no auxílio ao diagnóstico, na educação de pacientes e na otimização de fluxos de trabalho clínico. As melhores práticas envolvem o uso de instruções específicas, a definição de papéis (Role-Playing), a aplicação de técnicas de raciocínio passo a passo (Chain-of-Thought) e a garantia de que o contexto de domínio oftalmológico seja explicitamente fornecido. A pesquisa recente (2023-2025) demonstra que a engenharia de prompt pode aumentar significativamente a precisão dos LLMs em tarefas clínicas, como a triagem de doenças oculares, mas exige atenção rigorosa à confidencialidade dos dados e ao equilíbrio entre precisão e usabilidade.

## Examples
```
1. **Diagnóstico Diferencial com CoT e Role-Playing:**
```
Aja como um oftalmologista especialista em retina. Um paciente de 65 anos, diabético tipo 2, apresenta perda de visão central progressiva no olho direito. O exame de fundo de olho revela microaneurismas e hemorragias em chama.
1. Liste 3 diagnósticos diferenciais mais prováveis.
2. Para cada diagnóstico, descreva o raciocínio clínico (Chain-of-Thought) que leva a essa conclusão.
3. Apresente o resultado em uma tabela com as colunas: Diagnóstico, Raciocínio, Próximo Passo Diagnóstico.
```

2. **Geração de Material Educativo com Especificadores:**
```
Crie um folheto informativo de 200 palavras sobre "Glaucoma de Ângulo Aberto" para um paciente com nível de escolaridade fundamental.
1. Use linguagem simples e evite jargões médicos.
2. Inclua uma seção sobre a importância da adesão ao tratamento.
3. Apresente o texto final formatado em parágrafos curtos.
```

3. **Prompt Iterativo para Coleta de Histórico (Powerful Prompt):**
```
Você é um assistente virtual de triagem oftalmológica. Eu sou o paciente. Eu tenho dor e vermelhidão no olho esquerdo.
Continue me fazendo perguntas sobre meus sintomas, histórico médico e fatores de risco (apenas uma pergunta por vez) até que você tenha informações suficientes para sugerir se devo procurar atendimento de emergência ou agendar uma consulta de rotina.
```

4. **Criação de Questões de Múltipla Escolha (Treinamento):**
```
Gere 5 questões de múltipla escolha de nível de residência médica sobre a fisiopatologia da Retinopatia Diabética Proliferativa.
1. Cada questão deve ter 4 opções e apenas 1 correta.
2. Inclua a resposta correta e uma breve justificativa para cada questão.
```

5. **Simulação de Cenário Cirúrgico (Role-Playing Avançado):**
```
Aja como um cirurgião oftalmologista experiente. Descreva o passo a passo da técnica de facoemulsificação para catarata em um paciente com câmara anterior rasa.
1. Destaque os 3 pontos críticos de atenção.
2. Use termos técnicos apropriados.
```

6. **Otimização de Protocolo Clínico (Feedback Loop):**
```
Aqui está o nosso protocolo atual para acompanhamento de pacientes pós-operatórios de transplante de córnea: [INSERIR PROTOCOLO AQUI].
1. Analise o protocolo e sugira 3 melhorias para otimizar o fluxo de trabalho e a segurança do paciente.
2. Com base nas suas sugestões, reescreva a seção "Frequência de Consultas" do protocolo.
```
```

## Best Practices
**Instruções Específicas:** Fornecer instruções detalhadas e contextuais (ex: "Resuma a epidemiologia, fisiopatologia, diagnóstico e tratamento de X").
**Uso de Especificadores:** Definir o formato e o nível de detalhe/complexidade (ex: "Forneça um resumo breve/detalhado", "Explique em nível de consultor", "Apresente em tabela/tópicos").
**Priming (Preparação do LLM):** Definir o papel e o formato de entrada/saída esperado (ex: "Eu vou fornecer X; eu quero que você me retorne Y").
**Prompts de Incerteza:** Incluir a instrução "Se você não tiver certeza da resposta, diga que não sabe" para reduzir a chance de alucinações.
**Chain-of-Thought (CoT):** Encorajar o LLM a pensar passo a passo, simulando o raciocínio clínico para aumentar a precisão e a transparência.
**Role-Playing (Simulação de Papel):** Instruir o LLM a assumir o papel de um especialista em oftalmologia (ex: "Aja como um especialista em retina experiente.").
**Contexto Específico de Domínio:** Garantir que o prompt inclua terminologia médica, sintomas do paciente e critérios de diagnóstico relevantes.

## Use Cases
**Assistência ao Diagnóstico:** Auxiliar na triagem e diagnóstico diferencial de condições oftalmológicas (ex: Doença do Olho Seco, retinopatia diabética).
**Educação do Paciente:** Geração de materiais educativos claros e adaptados ao nível de compreensão do paciente.
**Geração de Perguntas de Múltipla Escolha:** Criação de questões de alta qualidade para treinamento e avaliação de residentes e estudantes.
**Otimização de Fluxos de Trabalho Clínico:** Gerenciamento da carga cognitiva associada a *checklists* e auxílio na identificação de erros médicos.
**Análise de Notas Clínicas:** Identificação precisa de componentes do exame oftalmológico a partir de notas de progresso.

## Pitfalls
**Confidencialidade de Dados:** **NUNCA** compartilhar dados confidenciais, mesmo desidentificados, com LLMs online.
**Viés de Memória:** O LLM pode ser influenciado por conversas anteriores na mesma sessão. Recomenda-se iniciar uma nova sessão para conversas não relacionadas.
**Alucinações:** A precisão pode ser comprometida se o prompt não for de alta qualidade, levando a respostas incorretas ou inventadas.
**Omissão de Raciocínio:** Prompts que pedem ao LLM para omitir o raciocínio podem resultar em saídas de qualidade inferior.
**Trade-off entre Precisão e Satisfação do Usuário:** Prompts mais complexos (como CoT) podem aumentar a precisão, mas também o tempo de resposta, afetando a satisfação do usuário.

## URL
[https://www.nature.com/articles/s41433-023-02772-w](https://www.nature.com/articles/s41433-023-02772-w)
