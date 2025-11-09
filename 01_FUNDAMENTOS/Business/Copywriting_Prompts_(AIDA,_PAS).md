# Copywriting Prompts (AIDA, PAS)

## Description
**Copywriting Prompts (AIDA, PAS)** é uma técnica de Prompt Engineering que utiliza frameworks clássicos de escrita persuasiva para estruturar a saída de modelos de linguagem (LLMs) como o ChatGPT, Gemini ou Claude. O objetivo é guiar a IA para produzir conteúdo que siga uma sequência lógica e psicológica comprovada, maximizando a conversão e o engajamento.

**AIDA (Atenção, Interesse, Desejo, Ação):** É o modelo mais tradicional, focado na jornada do consumidor.
1.  **Atenção (Attention):** Capturar o olhar do leitor (geralmente com uma manchete ou gancho forte).
2.  **Interesse (Interest):** Manter o leitor engajado, apresentando fatos relevantes e benefícios.
3.  **Desejo (Desire):** Criar uma conexão emocional, fazendo o leitor querer o produto/serviço.
4.  **Ação (Action):** Instruir o leitor sobre o próximo passo (Call to Action - CTA).

**PAS (Problema, Agitação, Solução):** É um modelo mais direto, focado em resolver uma dor específica.
1.  **Problema (Problem):** Identificar e declarar claramente a dor ou o desafio do público-alvo.
2.  **Agitação (Agitate):** Aumentar a intensidade da dor, descrevendo as consequências negativas de não resolver o problema.
3.  **Solução (Solution):** Apresentar o produto ou serviço como a solução ideal e imediata para a dor.

Ao incorporar AIDA ou PAS no prompt, o usuário está fornecendo uma **estrutura cognitiva** para a IA, transformando um pedido genérico em uma instrução de alto nível para a criação de copy persuasiva e orientada a resultados.

## Examples
```
**Exemplo 1: AIDA para Lançamento de Produto (E-mail)**

```
**Aja como um copywriter sênior.** Crie um e-mail de lançamento usando o framework AIDA para o nosso novo curso online: "Mestres do Prompt: Engenharia de Prompts para Negócios". O público-alvo são empreendedores digitais sobrecarregados.

**Atenção:** Manchete que destaque a frustração de prompts ruins.
**Interesse:** Apresente a Engenharia de Prompts como a nova habilidade essencial.
**Desejo:** Liste 3 benefícios de economizar 10 horas por semana com prompts otimizados.
**Ação:** CTA para se inscrever na lista de espera com 20% de desconto.
```

**Exemplo 2: PAS para Anúncio de Facebook (Dor Crônica)**

```
**Estrutura:** PAS (Problema, Agitação, Solução).
**Produto:** Cadeira ergonômica "AuraFlex".
**Público:** Profissionais que trabalham 8+ horas por dia em casa.
**Tom:** Empático e urgente.

**Problema:** Qual é a dor mais comum e ignorada do trabalho remoto?
**Agitação:** Descreva o custo a longo prazo dessa dor (médicos, perda de produtividade).
**Solução:** Apresente a AuraFlex como o investimento em saúde e produtividade. O texto final deve ter no máximo 150 palavras.
```

**Exemplo 3: AIDA para Landing Page (Geração de Leads)**

```
**Objetivo:** Gerar leads para um e-book gratuito sobre "Finanças Pessoais para Geração Z".
**Framework:** AIDA.
**Atenção:** Título que chame a atenção sobre a falta de educação financeira.
**Interesse:** Parágrafo sobre como a Geração Z pode começar a investir com pouco.
**Desejo:** 3 bullet points sobre o que o leitor aprenderá no e-book (ex: cortar dívidas, investir em cripto, aposentadoria precoce).
**Ação:** Botão CTA: "Baixe o E-book Gratuito Agora".
```

**Exemplo 4: PAS para Postagem no LinkedIn (B2B)**

```
**Aja como um consultor de TI.** Crie uma postagem no LinkedIn usando o framework PAS para promover nosso serviço de migração para a Nuvem.

**Problema:** O custo oculto de manter servidores legados.
**Agitação:** O risco de segurança e a lentidão que afetam a competitividade.
**Solução:** Nossa migração de Nuvem em 7 dias, com garantia de 99,9% de uptime.
**Instrução Adicional:** Use uma linguagem formal e inclua a hashtag #CloudMigration.
```

**Exemplo 5: AIDA Segmentado (Headline e CTA)**

```
**Tarefa:** Gerar apenas a seção "Atenção" e "Ação" do framework AIDA.
**Produto:** Software de automação de marketing para pequenas empresas.
**Público:** Donos de pequenos negócios sem tempo para marketing.

**Atenção (5 opções de Headline):** Foco em "tempo" e "automação".
**Ação (3 opções de CTA):** Foco em "teste gratuito" e "facilidade".
```
```

## Best Practices
**1. Forneça Contexto Detalhado (O "Briefing" do Prompt):** O prompt deve ser um briefing completo. Inclua a persona do público-alvo, o tom de voz (ex: urgente, empático, autoritário), o produto/serviço, e o objetivo final (ex: clique, venda, inscrição). **2. Segmente o Framework:** Em vez de pedir o texto inteiro de uma vez, peça à IA para gerar cada etapa do framework (Atenção, Interesse, Desejo, Ação ou Problema, Agitação, Solução) separadamente. Isso permite refinar cada parte antes de uni-las. **3. Use a Estrutura de "Stacking":** Combine o framework de copywriting (AIDA/PAS) com outras técnicas de prompt engineering, como a Persona (ex: "Aja como um copywriter sênior..."), ou o Chain-of-Thought (ex: "Primeiro, identifique a dor. Segundo, agite-a. Terceiro, apresente a solução."). **4. Iteração Humana (Human-in-the-Loop):** O texto gerado pela IA é um rascunho de alta qualidade. Sempre revise, edite e injete a voz humana e a prova social específica do seu negócio. A IA acelera, mas o toque final é seu. **5. Especifique o Formato de Saída:** Peça o output no formato ideal para o seu canal (ex: "Gere um tweet com 280 caracteres", "Gere um script de vídeo de 30 segundos", "Gere um parágrafo para landing page").

## Use Cases
**1. Criação de Anúncios de Alta Conversão:** Geração rápida de múltiplas variações de anúncios (Google Ads, Facebook Ads) que seguem a estrutura AIDA ou PAS, permitindo testes A/B eficientes. **2. Rascunho de Landing Pages:** Estruturação de páginas de destino, onde o AIDA é ideal para a sequência de convencimento e o PAS é excelente para ofertas de produtos que resolvem uma dor específica (SaaS, produtos de saúde). **3. Sequências de E-mail Marketing:** Criação de e-mails que guiam o lead através da jornada (AIDA) ou que focam em reengajamento ao agitar um problema não resolvido (PAS). **4. Roteiros de Vídeo Curto (TikTok, Reels):** O PAS é particularmente eficaz para vídeos curtos, onde o "Problema" e a "Agitação" ocorrem nos primeiros segundos para prender a atenção, e a "Solução" é o produto ou serviço. **5. Otimização de Títulos e Descrições de Produtos (E-commerce):** Uso do AIDA para criar títulos que chamam a atenção e descrições que geram desejo, especialmente em plataformas com limite de caracteres. **6. Geração de Conteúdo B2B (LinkedIn, Artigos):** O PAS é a estrutura preferida para conteúdo B2B, pois foca em problemas de negócios (Problema), demonstra o custo da inação (Agitação) e apresenta a solução corporativa (Solução).

## Pitfalls
**1. Confundir o Framework com o Output Final:** O erro mais comum é tratar o output da IA como a cópia final. A IA gera um rascunho estruturado; o copywriter humano deve sempre revisar, editar e injetar a voz da marca e a prova social real. **2. Prompts Genéricos:** Pedir apenas "Crie um AIDA para meu produto" resultará em um texto genérico e ineficaz. É crucial fornecer detalhes sobre o público, o tom, o produto e o objetivo de cada etapa do framework. **3. Agitação Insuficiente (no PAS):** No modelo PAS, a Agitação é a parte mais importante. Se o prompt não instruir a IA a aprofundar a dor e as consequências, a Solução não terá o impacto necessário. **4. Falta de Clareza na Ação (no AIDA):** O CTA (Ação) deve ser claro e único. Se o prompt resultar em múltiplos CTAs ou um CTA vago (ex: "Saiba mais"), a taxa de conversão será baixa. **5. Ignorar a Iteração:** O primeiro resultado raramente é o melhor. Falhar em refinar o prompt (ex: "Reescreva a seção Desejo com um tom mais luxuoso") é desperdiçar o potencial da IA. **6. Confiança Excessiva em Dados Antigos:** A IA pode basear o "Problema" ou "Interesse" em dados desatualizados. Sempre verifique a relevância e a precisão das informações factuais geradas.

## URL
[https://www.aidocmaker.com/blog/from-aida-to-pas-a-practical-guide-to-nailing-classic-copywriting-formulas-with-an-ai-word-generator](https://www.aidocmaker.com/blog/from-aida-to-pas-a-practical-guide-to-nailing-classic-copywriting-formulas-with-an-ai-word-generator)
