# Charades (Activity Recognition)

## Description
O Charades é um dataset de grande escala composto por 9.848 vídeos de atividades diárias internas, coletados via Amazon Mechanical Turk. O objetivo principal é guiar a pesquisa em reconhecimento de atividades em vídeo não estruturado e raciocínio de senso comum para atividades humanas diárias. Os vídeos são de curta duração (média de 30 segundos) e mostram 267 usuários diferentes encenando frases que incluem objetos e ações de um vocabulário fixo. O dataset é frequentemente usado como um benchmark para tarefas de reconhecimento e localização temporal de ações.

## Statistics
- **Vídeos:** 9.848 vídeos de atividades diárias internas.
- **Anotações Temporais:** 66.500 anotações para 157 classes de ações.
- **Rótulos de Objetos:** 41.104 rótulos para 46 classes de objetos.
- **Descrições Textuais:** 27.847 descrições textuais.
- **Tamanho Total:** O conjunto de dados em tamanho original é de aproximadamente 55 GB (vídeos), com a versão escalada para 480p em 13 GB.
- **Versões:** A versão original (v1) foi lançada em 2016. Extensões notáveis incluem Charades-Ego e Charades-STA.

## Features
- **Vídeos de Atividades Diárias:** 9.848 vídeos de atividades internas cotidianas.
- **Anotações Ricas:** Inclui 66.500 anotações temporais para 157 classes de ações, 41.104 rótulos para 46 classes de objetos e 27.847 descrições textuais.
- **Natureza Multi-rótulo:** As atividades podem ocorrer simultaneamente ou sequencialmente, tornando-o ideal para o reconhecimento de atividades multi-rótulo.
- **Diversidade de Dados:** Coletado de 267 usuários diferentes, garantindo uma variedade de cenários e estilos de atuação.
- **Variações:** Possui extensões como Charades-Ego (vídeos em primeira e terceira pessoa) e Charades-STA (para localização temporal de atividades com sentenças).

## Use Cases
- **Reconhecimento de Atividades em Vídeo (HAR):** Principal caso de uso, focado em identificar ações e atividades humanas em vídeos não estruturados.
- **Localização Temporal de Ações:** Utilizado para determinar o início e o fim exatos de uma atividade dentro de um vídeo.
- **Reconhecimento de Atividades Multi-rótulo:** Ideal para modelos que precisam identificar múltiplas ações ocorrendo simultaneamente ou sequencialmente.
- **Raciocínio de Senso Comum:** Pesquisa sobre a compreensão de como objetos e ações se relacionam em cenários cotidianos.
- **Visão Egocêntrica:** A extensão Charades-Ego é usada para treinar modelos que compreendem atividades a partir da perspectiva de primeira pessoa.
- **Geração de Legendas de Vídeo:** As descrições textuais e anotações temporais suportam o desenvolvimento de modelos de *video captioning*.

## Integration
O dataset Charades pode ser acessado e baixado diretamente da página oficial do Allen AI (prior.allenai.org/projects/charades) ou através de plataformas como o Hugging Face Datasets.

**Opções de Download:**
1.  **Página Oficial (Allen AI):** Oferece diversas opções de download, incluindo:
    *   Dados (escalados para 480p, 13 GB)
    *   Dados (tamanho original, 55 GB)
    *   Frames RGB e Optical Flow
    *   Anotações e Código de Avaliação (3 MB)
2.  **Hugging Face Datasets:** Pode ser carregado diretamente em ambientes Python usando a biblioteca `datasets`:
    ```python
    from datasets import load_dataset
    # Para a versão Charades-STA
    dataset = load_dataset("HuggingFaceM4/charades")
    ```
3.  **Repositórios GitHub:** Códigos de inicialização e algoritmos de linha de base estão disponíveis em repositórios como `gsig/charades-algorithms` para auxiliar na integração com frameworks como PyTorch e Torch.

**Instruções de Uso:**
Após o download, as anotações (em formato CSV) e os vídeos devem ser processados. O código de avaliação e os scripts de linha de base fornecidos pelos autores são essenciais para configurar o ambiente e executar modelos de reconhecimento de atividade. É recomendável começar com a versão escalada (480p) para testes iniciais.

## URL
[https://prior.allenai.org/projects/charades](https://prior.allenai.org/projects/charades)
