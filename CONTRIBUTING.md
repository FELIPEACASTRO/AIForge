# 🤝 Contributing Guide for AIForge

---

Thank you for your interest in contributing to **AIForge - The Ultimate Curated Collection of AI, Machine Learning, and Deep Learning Resources**! Your help is essential to keeping this collection as complete and up-to-date as possible.

---

## 🚀 How to Contribute

### 1. Find a Resource to Add

AIForge is an exhaustive collection of resources. We are looking for:

- **AI Models**: GitHub repositories, papers with code, models on Hugging Face.
- **Datasets**: Public, well-documented, and useful for the community.
- **Tools**: Libraries, frameworks, MLOps platforms.
- **Learning Resources**: Tutorials, courses, books, articles.
- **Niche Applications**: Special focus on **Finance, E-commerce, and Marketing**.

---

### 2. Check if the Resource Already Exists

Use GitHub search to ensure the resource hasn't been added yet:

```
repo:FELIPEACASTRO/AIForge "RESOURCE_NAME"
```

---

### 3. Add the Resource

1. **Fork** the repository.
2. **Clone** your fork locally.
3. **Create a branch** for your contribution:
   ```bash
   git checkout -b feature/AddResourceX
   ```
4. **Find the correct directory** for your resource, following the new Use Case structure:
   - **01_FUNDAMENTOS/**: Theory, Algorithms, Prompt Engineering, Learning Resources.
   - **02_MODELOS/**: LLMs, Vision, Audio, Multimodal Models, Architectures.
   - **03_DADOS_E_RECURSOS/**: Datasets, Data Tools, APIs, Vector Databases, MLOps.
   - **04_PRODUCAO_E_DEPLOY/**: Deployment, Optimization, Serving, Infrastructure, MLOps.
5. **Add the link** in Markdown format, following the existing pattern.

---

### 4. Follow the Standard Format

Format example:

```markdown
- [**Resource Name**](RESOURCE_URL) - Brief description of the resource.
```

**Real example:**

```markdown
- [**OpenHands**](https://github.com/OpenHands/OpenHands) - AI software engineer that writes code, creates features, and fixes bugs autonomously.
```

---

### 5. Commit and Push

```bash
git add .
git commit -m "feat: Add [Resource Name] in [Category]"
git push origin feature/AddResourceX
```

---

### 6. Open a Pull Request

- Go to your fork's page on GitHub.
- Click "Compare & pull request".
- Describe your contribution.
- Wait for review.

---

## ✅ Acceptance Criteria

- **Relevance**: The resource must be relevant to the AI community, focusing on **Deep Learning, Transfer Learning, and Production AI**.
- **Quality**: Must be a high-quality, well-documented, and functional resource.
- **Focus**: Priority for resources that fit the new logical structure. Resources related to **05_APLICACOES** (Projects/Applications) are highly encouraged.
- **Format**: Must follow the contribution format.

---

## 📜 Code of Conduct

Be respectful and constructive in all interactions. Follow the [Code of Conduct](./CODE_OF_CONDUCT.md).

---

## ❓ Questions

If you have any questions, open an [issue](https://github.com/FELIPEACASTRO/AIForge/issues).

---

Thank you for your contribution! 🎉
