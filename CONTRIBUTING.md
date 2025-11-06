# 🤝 Guia de Contribuição para AIForge

Obrigado por seu interesse em contribuir para o **AIForge - The Ultimate Curated Collection of AI, Machine Learning, and Deep Learning Resources**! Sua ajuda é fundamental para manter esta coleção a mais completa e atualizada possível.

## 🚀 Como Contribuir

### 1. Encontre um Recurso para Adicionar

O AIForge é uma coleção exaustiva de recursos. Buscamos:

- **Modelos de IA:** Repositórios GitHub, papers com código, modelos no Hugging Face.
- **Datasets:** Públicos, bem documentados e úteis para a comunidade.
- **Ferramentas:** Bibliotecas, frameworks, plataformas de MLOps.
- **Recursos de Aprendizagem:** Tutoriais, cursos, livros, artigos.
- **Aplicações de Nicho:** Foco especial em **Finanças, E-commerce e Marketing**.

### 2. Verifique se o Recurso já Existe

Use a busca do GitHub para garantir que o recurso ainda não foi adicionado:

```
repo:FELIPEACASTRO/AIForge "NOME_DO_RECURSO"
```

### 3. Adicione o Recurso

1. **Fork** o repositório.
2. **Clone** seu fork localmente.
3. **Crie uma branch** para sua contribuição:
   ```bash
   git checkout -b feature/AdicionarRecursoX
   ```
4. **Encontre o diretório correto** para o seu recurso, seguindo a nova estrutura por Caso de Uso:
   - **01\_LEARN/**: Cursos, livros, comunidades.
   - **02\_BUILD/**: Frameworks, modelos, datasets.
   - **03\_DEPLOY/**: MLOps, serving, infraestrutura.
   - **04\_APPLY/**: Aplicações em domínios específicos (Finanças, Saúde, etc.).
5. **Adicione o link** no formato Markdown, seguindo o padrão existente.

### 4. Siga o Formato Padrão

**Exemplo de formato:**

```markdown
- [**Nome do Recurso**](URL_DO_RECURSO) - Breve descrição do recurso (em português).
```

**Exemplo real:**

```markdown
- [**OpenHands**](https://github.com/OpenHands/OpenHands) - Engenheiro de software de IA que escreve código, cria features e resolve bugs autonomamente.
```

### 5. Faça Commit e Push

```bash
git add .
git commit -m "feat: Adiciona [Nome do Recurso] em [Categoria]"
git push origin feature/AdicionarRecursoX
```

### 6. Abra um Pull Request

- Vá para a página do seu fork no GitHub.
- Clique em "Compare & pull request".
- Descreva sua contribuição.
- Aguarde a revisão.

## ✅ Critérios de Aceitação

- **Relevância:** O recurso deve ser relevante para a comunidade de IA.
- **Qualidade:** Deve ser um recurso de alta qualidade, bem documentado e funcional.
- **Foco:** Prioridade para recursos que se encaixam na nova estrutura por Caso de Uso.
- **Formato:** Deve seguir o formato de contribuição.

## 📜 Código de Conduta

Seja respeitoso e construtivo em todas as interações. Siga o [Código de Conduta](./CODE_OF_CONDUCT.md).

## ❓ Dúvidas

Se tiver alguma dúvida, abra uma [issue](https://github.com/FELIPEACASTRO/AIForge/issues).

Obrigado por sua contribuição! 🎉
