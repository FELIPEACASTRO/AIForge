# Naive Bayes

> A family of fast, probabilistic classifiers that apply Bayes' theorem under the "naive" assumption that features are conditionally independent given the class.

## Why it matters

Naive Bayes is one of the simplest and fastest classifiers available: training and prediction are linear in the number of examples and features, it needs little data to estimate parameters, and it handles high-dimensional sparse inputs (like bag-of-words) gracefully. Despite its strong, usually-violated independence assumption, it is a remarkably strong baseline for text classification, spam filtering, and sentiment analysis, and is often the first model to try on a new categorical or count-based problem.

## Core concepts

**Bayes' theorem.** For a class `y` and feature vector `x = (x_1, ..., x_n)`:

`P(y | x) = P(y) * P(x | y) / P(x)`

**The naive assumption.** Features are assumed conditionally independent given the class, so `P(x | y) = prod_i P(x_i | y)`. The denominator `P(x)` is constant across classes, so the **MAP (maximum a posteriori)** decision rule is:

`y_hat = argmax_y P(y) * prod_i P(x_i | y)`

**Log-space computation.** To avoid floating-point underflow from multiplying many small probabilities, scores are computed as sums of logs:

`y_hat = argmax_y [ log P(y) + sum_i log P(x_i | y) ]`

**Priors `P(y)`** are usually estimated as class frequencies in the training set (MLE), or set uniform.

**Likelihoods `P(x_i | y)`** depend on the assumed distribution of features — this choice defines the *variant* (Gaussian, Multinomial, Bernoulli, etc.).

**Smoothing.** For discrete models, unseen feature/class combinations yield zero probability, which zeroes the whole product. **Laplace / Lidstone (additive) smoothing** adds a constant `alpha` (alpha = 1 = Laplace) to each count to avoid this.

**Why it works despite wrong assumptions.** Domingos & Pazzani (1997) showed the classifier can be optimal under zero-one loss even when independence is heavily violated, because correct *ranking* of the argmax matters more than calibrated probabilities. Note that probability *estimates* themselves are often poorly calibrated (pushed toward 0 or 1).

**Generative model.** Naive Bayes is generative — it models `P(x, y)` — which contrasts with discriminative models like logistic regression that model `P(y | x)` directly. The two form a classic generative/discriminative pair.

## Variants

| Variant | Feature type | Likelihood `P(x_i \| y)` | Typical use |
|---|---|---|---|
| **Gaussian NB** | Continuous | Normal: `N(mu_{iy}, sigma_{iy}^2)` per feature/class | Real-valued tabular features |
| **Multinomial NB** | Counts / frequencies | Multinomial over token counts | Text classification with word counts / tf-idf |
| **Bernoulli NB** | Binary (presence/absence) | Bernoulli per feature; penalizes non-occurring features | Short text, binary feature vectors |
| **Categorical NB** | Discrete categories | Categorical distribution per feature | Categorical tabular data |
| **Complement NB (CNB)** | Counts | Estimates per-class weights from the *complement* of each class | Imbalanced text; often beats Multinomial NB |
| **Flexible / KDE NB** | Continuous | Kernel density estimate instead of a single Gaussian | Non-Gaussian continuous features |

For text, the two canonical "event models" are the **multivariate Bernoulli** (word present/absent) and the **multinomial** (word counts); McCallum & Nigam (1998) found multinomial typically wins at larger vocabularies.

## Tools & libraries

| Tool | What it offers | URL |
|---|---|---|
| scikit-learn `naive_bayes` | GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB | https://scikit-learn.org/stable/modules/naive_bayes.html |
| scikit-learn `CountVectorizer` / `TfidfVectorizer` | Text → count/tf-idf feature matrices for NB | https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction |
| NLTK `NaiveBayesClassifier` | Educational NB with `show_most_informative_features` | https://www.nltk.org/api/nltk.classify.naivebayes.html |
| Apache Spark MLlib `NaiveBayes` | Distributed Multinomial/Bernoulli/Gaussian/Complement NB | https://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes |
| Weka `NaiveBayes` | GUI/Java NB with kernel-density and discretization options | https://weka.sourceforge.io/doc.dev/weka/classifiers/bayes/NaiveBayes.html |
| R `e1071::naiveBayes` | Classic R implementation (Gaussian + categorical) | https://cran.r-project.org/web/packages/e1071/ |
| `naivebayes` (R) | High-performance Gaussian/Multinomial/Bernoulli/Poisson NB | https://cran.r-project.org/web/packages/naivebayes/ |

## Learning resources

| Resource | Type | Link |
|---|---|---|
| ISLR — *An Introduction to Statistical Learning* (Naive Bayes, §4.4) | Book (free PDF) | https://www.statlearning.com/ |
| Murphy — *Probabilistic Machine Learning: An Introduction* (Naive Bayes ch.) | Book (free PDF) | https://probml.github.io/pml-book/book1.html |
| Hastie, Tibshirani & Friedman — *Elements of Statistical Learning* | Book (free PDF) | https://hastie.su.domains/ElemStatLearn/ |
| Manning, Raghavan & Schütze — *IR Book*, Ch. 13 "Text classification & Naive Bayes" | Textbook (free) | https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-1.html |
| Jurafsky & Martin — *Speech and Language Processing*, Ch. "Naive Bayes & Sentiment" | Textbook (free draft) | https://web.stanford.edu/~jurafsky/slp3/ |
| scikit-learn — Working With Text Data tutorial | Hands-on tutorial | https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html |
| StatQuest — Naive Bayes / Gaussian Naive Bayes | Video | https://www.youtube.com/watch?v=O2L2Uv9pdDA |

## Key papers

| Year | Paper | Why it matters | Link |
|---|---|---|---|
| 1997 | Domingos & Pazzani — *On the Optimality of the Simple Bayesian Classifier under Zero-One Loss* | Explains why NB works despite violated independence | https://link.springer.com/article/10.1023/A:1007413511361 |
| 1998 | McCallum & Nigam — *A Comparison of Event Models for Naive Bayes Text Classification* (AAAI Workshop) | Defines Bernoulli vs. multinomial text event models | https://aaai.org/papers/041-ws98-05-007/ |
| 2001 | Lewis — *Naive (Bayes) at Forty: The Independence Assumption in Information Retrieval* (ECML 1998, repr.) | Historical/IR perspective on the independence assumption | https://link.springer.com/chapter/10.1007/BFb0026666 |
| 2002 | Pang, Lee & Vaithyanathan — *Thumbs up? Sentiment Classification using Machine Learning Techniques* | Early NB vs. SVM/MaxEnt sentiment benchmark | https://arxiv.org/abs/cs/0205070 |
| 2003 | Rennie, Shih, Teevan & Karger — *Tackling the Poor Assumptions of Naive Bayes Text Classifiers* (ICML) | Introduces Complement NB + tf-idf-style fixes | https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf |
| 2004 | Metsis, Androutsopoulos & Paliouras — *Spam Filtering with Naive Bayes — Which Naive Bayes?* | Compares NB event models for spam | https://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf |

## Cross-references in AIForge

- [Machine_Learning](../../Machine_Learning/) — broader supervised-learning context
- [Bayesian_and_Probabilistic_ML](../../Bayesian_and_Probabilistic_ML/) — probabilistic modeling and priors
- [Discriminant_Analysis](../Discriminant_Analysis/) — LDA/QDA, the Gaussian generative siblings
- [Model_Evaluation](../../Model_Evaluation/) — metrics, calibration, cross-validation

## Sources

- scikit-learn — 1.9. Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
- scikit-learn — GaussianNB / MultinomialNB / BernoulliNB / ComplementNB API docs: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html
- McCallum & Nigam (1998), AAAI: https://aaai.org/papers/041-ws98-05-007/
- Domingos & Pazzani (1997), Machine Learning 29: https://link.springer.com/article/10.1023/A:1007413511361
- Rennie et al. (2003), ICML: https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
- Manning, Raghavan & Schütze, IR Book Ch.13: https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-1.html
