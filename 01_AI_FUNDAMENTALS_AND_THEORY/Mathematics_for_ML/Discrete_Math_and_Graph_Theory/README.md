# Discrete Mathematics and Graph Theory

> The mathematics of countable, structured objects — sets, logic, counting, and graphs — that underpins algorithms, combinatorial optimization, and modern graph neural networks (GNNs).

## Why it matters

Discrete math is the native language of computation: algorithm correctness, complexity analysis, and data structures all rest on logic, sets, and recurrences. Graph theory in particular models relational data — molecules, social networks, knowledge graphs, road maps — and is the formal foundation for message-passing GNNs, whose expressive power is provably bounded by the Weisfeiler–Lehman isomorphism test. Combinatorics drives the counting arguments behind sampling, capacity, and generalization bounds.

## Core concepts

- **Sets, relations, functions.** Operations (∪, ∩, \, ×), equivalence relations and partitions, partial orders (posets), and function properties (injective/surjective/bijective). Cardinality and countability.
- **Logic & proof.** Propositional/predicate logic, quantifiers, and proof techniques: direct, contrapositive, contradiction, and **induction** (weak/strong/structural) — the backbone of algorithm correctness.
- **Combinatorics.** Rules of sum/product; permutations $P(n,k)=\frac{n!}{(n-k)!}$ and combinations $\binom{n}{k}=\frac{n!}{k!(n-k)!}$; binomial theorem; inclusion–exclusion; pigeonhole principle; generating functions; recurrences (e.g. solving $T(n)=aT(n/b)+f(n)$ via the Master Theorem).
- **Graphs (definitions).** A graph $G=(V,E)$ with $|V|=n$, $|E|=m$. Directed vs. undirected, weighted, multigraphs, bipartite, DAGs, trees, planar graphs. Degree sequences; handshaking lemma $\sum_{v}\deg(v)=2m$.
- **Graph representations.** Adjacency matrix $A\in\{0,1\}^{n\times n}$, adjacency list, incidence matrix. Degree matrix $D$ and the **graph Laplacian** $L=D-A$ (and normalized $L_{\text{sym}}=I-D^{-1/2}AD^{-1/2}$), whose spectrum encodes connectivity, cuts, and diffusion.
- **Connectivity & structure.** Paths, cycles, components, spanning trees, cut vertices/bridges, $k$-connectivity, strongly connected components (Tarjan/Kosaraju).
- **Spectral graph theory.** Eigenvalues/eigenvectors of $A$ and $L$; the number of zero eigenvalues of $L$ equals the number of connected components; the Fiedler value (algebraic connectivity) governs spectral clustering and underlies spectral graph convolutions.
- **Graph isomorphism & color refinement.** The 1-Weisfeiler–Lehman (1-WL) test iteratively hashes multisets of neighbor labels; it is a fast (incomplete) isomorphism heuristic and the theoretical ceiling for message-passing GNN expressiveness.
- **Number theory & discrete probability (adjacent).** Modular arithmetic, GCD/Euclid, hashing; discrete distributions and random graphs (Erdős–Rényi $G(n,p)$).

## Algorithms / Methods

| Algorithm | Problem | Complexity | Notes |
|---|---|---|---|
| BFS / DFS | Traversal, components, bipartiteness | $O(n+m)$ | Foundation for most graph algorithms |
| Dijkstra | Single-source shortest path (non-negative) | $O(m+n\log n)$ | Binary/Fibonacci heap |
| Bellman–Ford | Shortest path with negative edges | $O(nm)$ | Detects negative cycles |
| Floyd–Warshall | All-pairs shortest paths | $O(n^3)$ | DP over intermediate vertices |
| Kruskal / Prim | Minimum spanning tree | $O(m\log n)$ | Union-Find / priority queue |
| Topological sort | DAG ordering | $O(n+m)$ | Scheduling, dependency resolution |
| Tarjan / Kosaraju | Strongly connected components | $O(n+m)$ | |
| Edmonds–Karp / Dinic | Max-flow / min-cut | $O(nm^2)$ / $O(n^2 m)$ | Network flow duality |
| Hopcroft–Karp | Maximum bipartite matching | $O(m\sqrt{n})$ | |
| PageRank | Node centrality / ranking | $O(m)$ per iter | Power iteration on stochastic matrix |
| Weisfeiler–Lehman | Graph isomorphism heuristic / kernels | $O(hm)$, $h$ iters | Basis of WL graph kernels and the GNN bound |
| Spectral clustering | Community detection / partitioning | $O(n^3)$ (eig) | Uses Laplacian eigenvectors |

## Tools & libraries

| Tool | Use | URL |
|---|---|---|
| NetworkX | Pure-Python graph creation, analysis, algorithms | https://networkx.org/ |
| igraph (python-igraph) | Fast C-backed graph analysis & community detection | https://python-igraph.org/ |
| Stanford SNAP / SNAP.py | Large-scale network analysis | https://snap.stanford.edu/ |
| graph-tool | High-performance C++ graph analysis (Python API) | https://graph-tool.skewed.de/ |
| PyTorch Geometric (PyG) | GNNs / geometric deep learning on PyTorch | https://pytorch-geometric.readthedocs.io/ |
| Deep Graph Library (DGL) | Framework-agnostic GNN library | https://www.dgl.ai/ |
| SciPy (sparse.csgraph) | Sparse adjacency, shortest paths, components | https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html |
| SageMath | Computer algebra incl. graph theory & combinatorics | https://www.sagemath.org/ |
| OGB (Open Graph Benchmark) | Standardized graph ML datasets & evaluation | https://ogb.stanford.edu/ |
| Gephi | Interactive network visualization | https://gephi.org/ |

## Learning resources

- **Rosen, *Discrete Mathematics and Its Applications*** — the standard comprehensive textbook (logic, combinatorics, graphs). https://archive.org/details/discretemathemat00kenn
- **Bondy & Murty, *Graph Theory with Applications*** — classic graph theory text (out of print; free download). https://freecomputerbooks.com/Graph-Theory-with-Applications.html
- **Easley & Kleinberg, *Networks, Crowds, and Markets*** — free full pre-publication draft; networks across CS/econ/sociology. https://www.cs.cornell.edu/home/kleinber/networks-book/
- **MIT 6.042J *Mathematics for Computer Science*** — full OCW course with notes/videos. https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/
- **CLRS, *Introduction to Algorithms*** — definitive treatment of graph & combinatorial algorithms. https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/
- **Hamilton, *Graph Representation Learning*** (free book) — the bridge from graph theory to GNNs. https://www.cs.mcgill.ca/~wlh/grl_book/
- **Stanford CS224W *Machine Learning with Graphs*** — slides + lectures on graph ML. https://web.stanford.edu/class/cs224w/
- **Distill — *A Gentle Introduction to Graph Neural Networks*** — interactive visual primer. https://distill.pub/2021/gnn-intro/

## Key papers

- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (2016) — the GCN. https://arxiv.org/abs/1609.02907
- Hamilton, Ying & Leskovec, *Inductive Representation Learning on Large Graphs* (GraphSAGE, 2017). https://arxiv.org/abs/1706.02216
- Veličković et al., *Graph Attention Networks* (GAT, 2017). https://arxiv.org/abs/1710.10903
- Xu et al., *How Powerful are Graph Neural Networks?* (GIN; the WL expressiveness bound, 2018). https://arxiv.org/abs/1810.00826
- Wu et al., *A Comprehensive Survey on Graph Neural Networks* (2019). https://arxiv.org/abs/1901.00596
- *Foundations and Frontiers of Graph Learning Theory* (2024) — expressiveness, generalization, over-smoothing/over-squashing. https://arxiv.org/abs/2407.03125

## Cross-references in AIForge

- [Linear Algebra](../Linear_Algebra/) — adjacency/Laplacian matrices, eigendecomposition, spectral methods
- [Probability Theory](../Probability_Theory/) — random graphs, sampling, discrete distributions
- [Convex Optimization](../Convex_Optimization/) — combinatorial & network-flow problems as optimization
- [Deep Learning](../../Deep_Learning/) — graph neural networks and message passing
- [Knowledge Graphs](../../Knowledge_Graphs/) — graph-structured knowledge representation

## Sources

- arXiv: https://arxiv.org/abs/1609.02907 · https://arxiv.org/abs/1706.02216 · https://arxiv.org/abs/1710.10903 · https://arxiv.org/abs/1810.00826 · https://arxiv.org/abs/1901.00596 · https://arxiv.org/abs/2407.03125
- NetworkX: https://networkx.org/ · python-igraph: https://python-igraph.org/ · PyG: https://pytorch-geometric.readthedocs.io/ · DGL: https://www.dgl.ai/ · OGB: https://ogb.stanford.edu/
- Easley & Kleinberg book: https://www.cs.cornell.edu/home/kleinber/networks-book/ · MIT 6.042J: https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/ · Stanford CS224W: https://web.stanford.edu/class/cs224w/ · Distill GNN intro: https://distill.pub/2021/gnn-intro/ · Hamilton GRL book: https://www.cs.mcgill.ca/~wlh/grl_book/
