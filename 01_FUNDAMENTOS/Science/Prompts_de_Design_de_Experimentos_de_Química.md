# Prompts de Design de Experimentos de Química

## Description
Técnica de Prompt Engineering focada em alavancar Large Language Models (LLMs) e agentes de IA para planejar, otimizar e, em alguns casos, executar experimentos químicos. A abordagem mais avançada envolve o uso de um agente de IA (como o ChemCrow, descrito na Nature Machine Intelligence) que utiliza um fluxo de trabalho iterativo (Pensamento, Ação, Observação - ReAct/MRKL) para interagir com ferramentas químicas especializadas e bases de dados. Isso transforma uma solicitação de texto simples em um plano de ação experimental detalhado, cobrindo desde a retrossíntese até a avaliação de segurança e otimização de rendimento.

## Examples
```
1. **Síntese de um Composto Específico:** "Como um agente químico autônomo com acesso a bases de dados de reações (e.g., Reaxys, SciFinder) e ferramentas de segurança (e.g., PubChem), planeje a síntese do 1-feniletanol a partir do benzeno. O plano deve incluir a rota de retrossíntese, as condições de reação para cada etapa (temperatura, solvente, catalisador) e uma avaliação de risco de segurança para o procedimento final."
2. **Otimização de Rendimento:** "O rendimento da reação de Suzuki-Miyaura entre o iodobenzeno e o ácido fenilborônico é de 75%. Use suas ferramentas de otimização (e.g., algoritmos de otimização bayesiana) para sugerir três modificações nas condições de reação (catalisador, ligante, solvente ou temperatura) que maximizem o rendimento para mais de 90%. Apresente as três melhores sugestões com a justificativa química e o rendimento teórico esperado."
3. **Descoberta de Novo Material:** "Projete um novo polímero condutor (material) para uso em células solares orgânicas (OPVs). O material deve ter um *band gap* inferior a 1.8 eV e ser solúvel em clorofórmio. Use suas ferramentas de modelagem molecular (e.g., DFT) para sugerir a estrutura molecular do monômero e o procedimento de síntese de polimerização."
4. **Análise de Segurança e Risco:** "Analise a reação de nitração do tolueno para a produção de TNT. Use suas ferramentas de segurança e termodinâmica para identificar os principais riscos (explosão, toxicidade, subprodutos) e sugira um protocolo de mitigação de risco detalhado, incluindo os equipamentos de proteção individual (EPIs) necessários e o procedimento de descarte de resíduos."
5. **Planejamento de Rota de Retrossíntese:** "Determine a rota de retrossíntese mais eficiente e econômica para o medicamento antiviral Remdesivir. Use suas ferramentas de busca de rotas (e.g., retrosynthesis LLMs) para comparar pelo menos duas rotas publicadas, avaliando o número de etapas, o custo estimado dos reagentes de partida e a toxicidade dos intermediários."
6. **Simulação e Previsão de Propriedades:** "Preveja o pKa do ácido acético em metanol a 25°C. Use suas ferramentas de simulação (e.g., COSMO-RS) para calcular o valor e compare-o com o valor experimental em água. Explique a diferença observada com base nos efeitos do solvente."
```

## Best Practices
- **Definição Clara do Objetivo:** O prompt deve ser específico sobre o alvo químico (molécula, reação, propriedade) e o resultado desejado (plano, otimização, previsão).
- **Especificação de Restrições:** Incluir restrições práticas como custo, segurança, reagentes disponíveis ou condições de laboratório (e.g., "sem usar metais pesados", "temperatura máxima de 80°C").
- **Integração Explícita de Ferramentas:** Mencionar as ferramentas ou bases de dados que o LLM deve usar (e.g., "Use PubChem para dados de segurança", "Consulte o ChemSpider para estruturas").
- **Uso do Formato ReAct/Chain-of-Thought:** Para tarefas complexas, instruir o LLM a seguir um processo de raciocínio lógico (Pensamento, Ação, Observação) antes de apresentar a resposta final.

## Use Cases
- **Síntese Orgânica e Inorgânica:** Planejamento de rotas de síntese para moléculas complexas.
- **Descoberta e Otimização de Medicamentos:** Sugestão de novos candidatos a fármacos e otimização de suas propriedades (ADMET).
- **Ciência de Materiais:** Design de novos materiais com propriedades específicas (e.g., polímeros, catalisadores, semicondutores).
- **Análise de Risco e Segurança:** Avaliação de perigos em reações químicas e desenvolvimento de protocolos de segurança.
- **Otimização de Processos:** Ajuste fino de condições de reação para maximizar rendimento, seletividade ou sustentabilidade.

## Pitfalls
- **Alucinações Químicas:** O LLM pode gerar rotas de reação ou moléculas que são termodinamicamente ou cineticamente inviáveis.
- **Dependência de Dados de Treinamento:** O modelo pode repetir erros ou vieses presentes nos dados de treinamento, especialmente em química de nicho.
- **Risco de Segurança Não Mitigado:** A falta de integração robusta com ferramentas de segurança pode levar à sugestão de procedimentos perigosos.
- **Superestimação da Capacidade:** O usuário pode superestimar a capacidade do LLM de substituir a intuição e o conhecimento de um químico experiente.

## URL
[https://www.nature.com/articles/s42256-024-00832-8](https://www.nature.com/articles/s42256-024-00832-8)
