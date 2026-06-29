# Association Rule Learning

> Unsupervised mining of frequent itemsets and "if-then" rules (`X ⇒ Y`) that capture co-occurrence patterns in transactional data — the engine behind classic market-basket analysis.

## Why it matters

Association rule learning surfaces actionable, human-readable patterns ("customers who buy X also buy Y") from large, sparse transaction logs without labels or distributional assumptions. It underpins recommendation, cross-sell/up-sell, store layout, web-usage mining, and bioinformatics (e.g., gene/protein co-occurrence). Because rules are interpretable and the algorithms scale to millions of transactions, the paradigm remains a staple of exploratory data mining decades after its introduction.

## Core concepts

Let `I = {i_1, ..., i_m}` be a set of **items** and let the database `D` be a collection of **transactions**, each a subset `T ⊆ I`. An **itemset** is any subset of `I`; a `k`-itemset has `k` items. A rule is an implication `X ⇒ Y` with `X, Y ⊆ I` and `X ∩ Y = ∅`.

- **Support**: `supp(X) = |{T ∈ D : X ⊆ T}| / |D|` — the fraction of transactions containing `X`. An itemset is **frequent** if `supp(X) ≥ minsup`.
- **Confidence**: `conf(X ⇒ Y) = supp(X ∪ Y) / supp(X)` — conditional probability `P(Y | X)`, the rule's reliability.
- **Lift**: `lift(X ⇒ Y) = supp(X ∪ Y) / (supp(X) · supp(Y)) = conf(X ⇒ Y) / supp(Y)`. Lift `> 1` ⇒ positive correlation, `= 1` ⇒ independence, `< 1` ⇒ negative.
- **Leverage**: `supp(X ∪ Y) − supp(X)·supp(Y)`; **Conviction**: `(1 − supp(Y)) / (1 − conf(X ⇒ Y))`; **Zhang's metric** for asymmetric association. These mitigate confidence's bias toward frequent consequents.

**Two-phase recipe.** (1) Find all frequent itemsets with `supp ≥ minsup`; (2) from each frequent itemset generate rules with `conf ≥ minconf`. Phase 1 is the hard part — the itemset lattice has `2^m` candidates.

**Downward-closure (Apriori) property**: every subset of a frequent itemset is frequent (equivalently, every superset of an infrequent itemset is infrequent). This monotonicity is what lets algorithms prune the lattice instead of enumerating it.

**Condensed representations** reduce rule explosion: a **closed** itemset has no superset with equal support (lossless w.r.t. support); a **maximal** itemset has no frequent superset (lossless only w.r.t. membership); **generators** are minimal subsets with a given support. Mining closed/maximal sets yields far fewer, non-redundant patterns.

## Algorithms / Methods

| Algorithm | Strategy | Data layout | Key idea | Notes |
|---|---|---|---|---|
| **Apriori** | Breadth-first, candidate generate-and-test | Horizontal (TID → items) | Level-wise `(k)→(k+1)` joins + downward-closure pruning | Simple, interpretable; many DB scans, costly candidate sets |
| **AprioriTID / AprioriHybrid** | Apriori with TID-set encoding | Hybrid | Replaces DB by candidate-TID lists in later passes | Faster on later passes than plain Apriori |
| **FP-Growth** | Depth-first, pattern-fragment growth | Compressed prefix tree (FP-tree) | No candidate generation; recursive conditional FP-trees | ~order of magnitude faster than Apriori; 2 DB scans |
| **Eclat** | Depth-first, equivalence-class lattice | Vertical (item → TID-list) | Support via TID-list **intersection**; diffsets (dEclat) | No hash trees, few scans; memory grows with TID-lists |
| **FP-Max / MAFIA / GenMax** | Maximal-itemset mining | FP-tree / vertical | Keep only maximal frequent itemsets | Smallest output, lossy beyond membership |
| **CHARM / Closet+ / AClose** | Closed-itemset mining | Vertical / FP-tree | Mine closed sets → non-redundant rule bases | Lossless support, compact rule set |
| **H-Mine / LCM** | Hyper-structure / linear-time closed | Array / vertical | LCM (Linear time Closed itemset Miner) is a strong baseline | LCM won FIMI workshops |

## Tools & libraries

| Tool | Language | What it offers | URL |
|---|---|---|---|
| **mlxtend** (`frequent_patterns`) | Python | `apriori`, `fpgrowth`, `fpmax`, `association_rules` over one-hot/sparse DataFrames | https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/ |
| **SPMF** | Java | 250+ pattern-mining algorithms (Apriori, FP-Growth, Eclat, CHARM, LCM, sequential rules) | https://www.philippe-fournier-viger.com/spmf/ |
| **arules** / **arulesViz** | R | Reference implementation of Apriori/Eclat + interactive rule visualization | https://cran.r-project.org/package=arules |
| **Weka** (`Apriori`, `FPGrowth`) | Java | GUI + API association mining over ARFF data | https://ml.cms.waikato.ac.nz/weka/ |
| **Orange** (Associate add-on) | Python/GUI | Visual workflow nodes for frequent itemsets & rules | https://orangedatamining.com/ |
| **efficient-apriori** | Python | Lightweight pure-Python Apriori, no DataFrame dependency | https://github.com/tommyod/Efficient-Apriori |
| **fim (PyFIM)** | Python/C | Fast Apriori/Eclat/FP-Growth bindings by C. Borgelt | https://borgelt.net/pyfim.html |
| **Spark MLlib** (`FPGrowth`, `PrefixSpan`) | Scala/Python | Distributed parallel FP-Growth for big data | https://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html |

## Learning resources

- **Han, Kamber & Pei — *Data Mining: Concepts and Techniques* (3rd ed., Morgan Kaufmann, 2011)**, Ch. 6–7 on frequent patterns, associations, and advanced pattern mining (the canonical textbook treatment). https://www.sciencedirect.com/book/9780123814791/data-mining-concepts-and-techniques
- **Tan, Steinbach, Karpatne & Kumar — *Introduction to Data Mining* (2nd ed.)**, Ch. 5 on association analysis; free chapter PDFs and slides. https://www-users.cse.umn.edu/~kumar001/dmbook/index.php
- **Zaki & Meira — *Data Mining and Machine Learning: Fundamental Concepts and Algorithms* (2nd ed., 2020)** — itemset/rule chapters available free. https://dataminingbook.info/
- **mlxtend tutorials** — worked Apriori and FP-Growth notebooks with one-hot encoding and metric thresholds. https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/ and https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
- **arules / arulesViz vignettes** — practical R walkthroughs incl. the `Groceries` market-basket dataset. https://cran.r-project.org/web/packages/arules/vignettes/arules.pdf

## Key papers

- Agrawal, Imieliński & Swami (1993), *Mining Association Rules between Sets of Items in Large Databases*, ACM SIGMOD — introduces support/confidence and the problem. https://doi.org/10.1145/170035.170072
- Agrawal & Srikant (1994), *Fast Algorithms for Mining Association Rules in Large Databases*, VLDB '94, pp. 487–499 — the **Apriori** algorithm and downward-closure pruning. https://www.vldb.org/conf/1994/P487.PDF
- Han, Pei & Yin (2000), *Mining Frequent Patterns without Candidate Generation*, ACM SIGMOD — **FP-Growth** and the FP-tree. https://doi.org/10.1145/342009.335372
- Han, Pei, Yin & Mao (2004), *Mining Frequent Patterns without Candidate Generation: A Frequent-Pattern Tree Approach*, Data Mining and Knowledge Discovery 8:53–87 — extended FP-Growth journal version. https://doi.org/10.1023/B:DAMI.0000005258.31418.83
- Zaki (2000), *Scalable Algorithms for Association Mining*, IEEE TKDE 12(3):372–390 — **Eclat** and vertical TID-list mining. https://doi.org/10.1109/69.846291
- Zaki & Hsiao (2002), *CHARM: An Efficient Algorithm for Closed Itemset Mining*, SIAM SDM — closed-itemset mining for non-redundant rules. https://doi.org/10.1137/1.9781611972726.27
- Brin, Motwani, Ullman & Tsur (1997), *Dynamic Itemset Counting and Implication Rules for Market Basket Data*, ACM SIGMOD — DIC and the **conviction** metric. https://doi.org/10.1145/253260.253325

## Cross-references in AIForge

- [Clustering Algorithms](../Clustering_Algorithms/) — sibling unsupervised pattern-discovery family.
- [Recommender Systems](../../Recommender_Systems/) — rules as a transparent baseline for cross-sell/up-sell.
- [Knowledge Graphs](../../Knowledge_Graphs/) — rule mining over relational/graph facts.
- [Statistical Learning](../../Statistical_Learning/) — significance, correlation, and multiple-testing context for rule metrics.

## Sources

- Apriori (VLDB '94): https://www.vldb.org/conf/1994/P487.PDF · https://dl.acm.org/doi/10.5555/645920.672836
- FP-Growth (SIGMOD 2000): https://doi.org/10.1145/342009.335372 · journal: https://link.springer.com/article/10.1023/B:DAMI.0000005258.31418.83
- Eclat / Scalable Algorithms for Association Mining (IEEE TKDE 2000): https://doi.org/10.1109/69.846291 · https://www.scirp.org/reference/referencespapers?referenceid=3413098
- mlxtend frequent_patterns docs: https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/
- SPMF library (JMLR 2014): https://jmlr.csail.mit.edu/papers/volume15/fournierviger14a/fournierviger14a.pdf · https://www.philippe-fournier-viger.com/spmf/
- arules (CRAN): https://cran.r-project.org/package=arules
- Han, Kamber & Pei textbook: https://www.sciencedirect.com/book/9780123814791/data-mining-concepts-and-techniques
- Tan et al., *Introduction to Data Mining*: https://www-users.cse.umn.edu/~kumar001/dmbook/index.php
