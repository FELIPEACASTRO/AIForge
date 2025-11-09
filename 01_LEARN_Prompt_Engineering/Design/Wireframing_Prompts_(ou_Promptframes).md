# Wireframing Prompts (ou Promptframes)

## Description
Wireframing Prompts, também conhecidos como **Promptframes**, representam uma evolução do tradicional wireframe de UX/UI, integrando a escrita de prompts de IA generativa ao processo de design. Um Promptframe é um entregável de design que documenta as metas e requisitos de conteúdo para prompts de IA, baseando-se no layout e na funcionalidade de um wireframe. O objetivo principal é utilizar a IA para gerar conteúdo de alta fidelidade e relevância para preencher os elementos de design (como títulos, textos de botões, descrições de produtos, etc.), substituindo o uso de *lorem ipsum* e acelerando a criação de protótipos realistas para testes de usuário. Essa técnica se posiciona entre a criação do wireframe de baixa fidelidade e o protótipo detalhado, garantindo que o conteúdo seja relevante desde as fases iniciais do projeto [1] [2].

## Examples
```
1.  **Wireframe de Página Inicial de E-commerce (Alta Fidelidade):**
    > "Projete um wireframe de desktop de alta fidelidade para a página inicial de uma loja virtual de tênis de corrida. Inclua um carrossel de herói com uma chamada para ação clara ('Compre a Nova Coleção'), uma seção de 'Produtos em Destaque' com 4 itens, e uma barra de navegação superior com 'Masculino', 'Feminino', 'Ofertas' e 'Carrinho'. Use um esquema de cores branco/cinza e tipografia limpa. O layout deve ser responsivo para mobile."
2.  **Wireframe de Formulário de Cadastro (Foco Funcional):**
    > "Crie um wireframe de baixa fidelidade para um formulário de cadastro de usuário. O formulário deve incluir campos para Nome Completo, E-mail, Senha (com confirmação), e uma caixa de seleção para 'Aceito os Termos de Serviço'. Adicione um botão 'Cadastrar'. Inclua anotações para o estado de erro de 'E-mail já cadastrado' e o estado de sucesso após o envio."
3.  **Wireframe de Painel de Controle (SaaS):**
    > "Desenvolva um wireframe de desktop para um painel de controle de gerenciamento de projetos SaaS. O layout deve ser composto por um menu lateral esquerdo fixo (com ícones para Dashboard, Tarefas, Membros e Configurações) e uma área de conteúdo principal. A área principal deve exibir um gráfico de Gantt simplificado e uma lista de 'Tarefas Pendentes'. O design deve ser minimalista e focado na usabilidade."
4.  **Wireframe de Aplicativo Móvel (Estado Vazio):**
    > "Gere um wireframe móvel para a tela 'Minhas Receitas' de um aplicativo de culinária. O wireframe deve focar no **estado vazio** da tela. Inclua um ícone ilustrativo, o texto 'Você ainda não salvou nenhuma receita' e um botão de chamada para ação 'Explorar Receitas'. O design deve ser amigável e encorajador."
5.  **Wireframe de Bloco de Conteúdo (Restrição de Layout):**
    > "Crie um bloco de conteúdo para a seção 'Nossos Valores' de um site institucional. O bloco deve ter um layout de três colunas, cada uma contendo um ícone, um título (máximo 5 palavras) e uma breve descrição (máximo 2 frases). Use um estilo de ícone linear e moderno. **Restrição:** Evite o uso de cores vibrantes, mantendo-se em tons de azul marinho e branco."
6.  **Wireframe de Checkout (Fluxo):**
    > "Projete o wireframe de uma única tela de checkout para um e-commerce. A tela deve consolidar as etapas de 'Informações de Envio', 'Método de Pagamento' e 'Resumo do Pedido'. Use um layout de coluna única e destaque o preço total final. Inclua um campo para 'Cupom de Desconto' e um botão 'Finalizar Compra'."
```

## Best Practices
*   **Seja Específico e Estruturado:** Defina claramente o propósito, o layout, a fidelidade (baixa ou alta) e os componentes funcionais. Quanto mais detalhes, melhor será o resultado inicial [2].
*   **Defina a Fidelidade:** Especifique se o objetivo é um esboço simples (*low-fidelity*) ou um design mais detalhado com tipografia e espaçamento (*high-fidelity*).
*   **Inclua Restrições:** Use restrições de design (ex: "Use grid de 8px", "Evite barras laterais") para guiar a IA e evitar *templates* genéricos [2].
*   **Pense em Acessibilidade:** Inclua requisitos de acessibilidade no prompt, como "Todos os elementos interativos devem atender aos padrões WCAG AA para contraste" [2].
*   **Iteração e Refinamento:** Use o output da IA como um ponto de partida. Combine variações e refine manualmente os elementos para alinhamento e usabilidade.

## Use Cases
*   **Aceleração de Protótipos:** Geração rápida de protótipos preenchidos com conteúdo realista para testes de usabilidade imediatos.
*   **Validação de Conteúdo:** Testar a eficácia de diferentes textos (títulos, CTAs) em um contexto de layout antes de investir em design de alta fidelidade.
*   **Documentação de Design:** Criar um artefato de design (*Promptframe*) que serve como ponte entre o wireframe e o protótipo, comunicando requisitos de conteúdo para *stakeholders* e desenvolvedores [1].
*   **Design de Estados de Borda:** Geração de wireframes para estados não ideais, como telas vazias, mensagens de erro e estados de carregamento.
*   **Exploração de Layouts:** Explorar rapidamente múltiplas variações de layout para uma mesma tela, apenas alterando as instruções do prompt.

## Pitfalls
*   **Templates Genéricos:** A IA pode gerar layouts convencionais e previsíveis. **Solução:** Use restrições e alimente a IA com exemplos de designs não convencionais [2].
*   **Ignorar Estados de Borda:** A IA tende a focar no fluxo ideal, ignorando estados vazios ou de erro. **Solução:** Solicite explicitamente o design desses estados no prompt [2].
*   **Problemas de Acessibilidade:** A IA pode negligenciar o contraste, a ordem de foco ou as dicas de texto alternativo. **Solução:** Defina regras de acessibilidade como requisitos obrigatórios no prompt [2].
*   **Perda de Racional:** A IA não explica o *porquê* de suas escolhas de design. **Solução:** Peça à IA para incluir anotações explicando a lógica por trás de cada decisão.
*   **Dependência Excessiva:** Tratar o output da IA como o design final, em vez de um ponto de partida que requer refinamento manual e validação do usuário.

## URL
[https://www.nngroup.com/articles/promptframes/](https://www.nngroup.com/articles/promptframes/)
