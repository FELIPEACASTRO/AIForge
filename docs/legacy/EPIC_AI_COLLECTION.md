# 🚀 EPIC AI COLLECTION - The Ultimate AI Resources Repository

## 📊 Overview

The most comprehensive collection of AI resources on GitHub, featuring **10,000+ curated resources** across algorithms, libraries, frameworks, models, and applications.

---

# PART 1: AI ALGORITHMS & LIBRARIES

## Ai Algorithms Classification

# AI_Algorithms_Classification

A comprehensive list of **200+ classification algorithms** for AI and ML with descriptions is not fully covered in commonly cited articles or textbooks—most public sources cite only the most widely used algorithms. However, I will present a detailed, organized list that covers the main algorithm **families**, providing individual named algorithms, variants, and frameworks as sub-items. This will maximize breadth and clarity for anyone seeking a reference or practical guide.

Given the scale requested, I group algorithms by **core categories** and enumerate as many distinctive members, subtypes, and notable open-source implementations as feasible. For well-established algorithm families (e.g., Decision Trees, Ensemble Methods), I list specific variants and extensions under their umbrella.

For each core family, the **algorithm name** is in bold, followed by a concise description.

---

## Decision Tree-Based Algorithms

- **Decision Tree (DT)**: Splits data recursively based on feature values to partition classes[1][2][3][4].
- **CART (Classification and Regression Trees)**: Binary splits using Gini index or entropy[3].
- **ID3 (Iterative Dichotomiser 3)**: Builds trees using information gain[3].
- **C4.5**: Successor of ID3 using gain ratio, handles missing values[3].
- **C5.0**: Improved C4.5, faster and memory efficient[3].
- **CHAID (Chi-squared Automatic Interaction Detection)**: Multivariate splits using Chi-square test[3].
- **Decision Stump**: Single-level decision tree—often used as a weak learner in ensembles[3].
- **Conditional Inference Trees**: Statistically rigorous method for splits[3].
- **M5**: Regression/classification tree with linear models at leaves[3].

## Random Forest Variants

- **Random Forest**: Ensemble of decorrelated decision trees—majority vote for class assignment[1][2][4][5].
- **Extra Trees (Extremely Randomized Trees)**: Like Random Forest with more randomization in split selection.
- **Oblique Random Forest**: Trees that split using linear combinations of features.
- **Rotation Forest**: Trees built on rotated features for diversity.

## Gradient Boosted Trees and Related Ensembles

- **Gradient Boosting Classifier**: Sequentially fits trees to correct predecessor errors[5].
- **XGBoost**: Efficient implementation, regularization, parallelism—very popular in ML competitions[5].
- **LightGBM**: Fast, leaf-wise histogram-based boosting, efficient for large datasets[5].
- **CatBoost**: Handles categorical features natively; symmetric trees for stability[5].
- **AdaBoost**: Adaptive boosting using weighted majority[5].
- **LogitBoost**: Boosting for logistic regression loss.
- **RUSBoost**: Boosting framework for imbalanced classes via random under-sampling.

## Bayesian Algorithms

- **Naive Bayes**: Assumes independent features, fast for text and other high-dimensional data[2][3][5].
- **Gaussian Naive Bayes**: Assumes normal distribution for features[3][5].
- **Multinomial Naive Bayes**: For count data (text classification especially)[3][5].
- **Bernoulli Naive Bayes**: For binary/boolean features[10].
- **Complement Naive Bayes**: Modifies Multinomial for imbalanced class problems.
- **Averaged One-Dependence Estimators (AODE)**: Relaxes independence assumption; averages all one-dependence classifiers[3].
- **Bayesian Belief Network (BBN)**: Graph-based probabilistic model[3].
- **Bayesian Network (BN)**: Probabilistic graphical model encoding dependencies[3].
- **Bayesian Logistic Regression**: Probabilistic extension of logistic regression.

## Linear Model-Based Methods

- **Logistic Regression**: Linear model for binary/multiclass classification via logistic (softmax) function[4][5].
- **Multinomial Logistic Regression**: Extends logistic regression to multiple classes.
- **Ridge Classifier**: Logistic regression with L2 regularization.
- **LASSO Logistic Regression**: L1 regularization for feature selection.
- **Elastic Net Classifier**: Combines L1 and L2 penalties.

## Support Vector Machines (SVM)

- **Support Vector Machine (SVM)**: Finds hyperplane maximizing margin between classes[1][2][5].
- **Linear SVM**: For linearly separable data.
- **Nonlinear SVM**: Kernel trick to allow nonlinear boundaries.
- **Nu-SVC**: Variant with Nu regularization parameter.
- **One-vs-One SVM**: Decomposes multi-class into one-vs-one binary problems.
- **One-vs-Rest SVM**: Decomposes multi-class into one-vs-all problems.

## Nearest Neighbor Methods

- **K-Nearest Neighbor (KNN)**: Assigns label based on majority vote among k closest examples[1][2][5].
- **Weighted KNN**: Closer neighbors have greater influence.
- **Radius Neighbors Classifier**: Uses neighbors within a given distance.
- **Condensed Nearest Neighbor**: Reduces memory usage, subsampling data points.
- **Edited/Reduced Nearest Neighbor**: Cleans data, removes noisy instances.

## Neural Network and Deep Learning Classifiers

- **Perceptron**: Single-layer neural unit for binary classification.
- **Multi-Layer Perceptron (MLP)**: Feedforward neural network for nonlinear separation.
- **Convolutional Neural Network (CNN)**: For spatial/image data.
- **Recurrent Neural Network (RNN)**: For sequential/time-series data.
- **Long Short-Term Memory (LSTM)**, **GRU**, etc.: RNN variants for sequence learning.
- **Capsule Network Classifier**: Encodes spatial hierarchies in images.
- **Self-Attention Classifier**: Utilizes attention mechanisms (e.g., Transformer-based).
- **Residual Network Classifier (ResNet)**: Deep networks with skip connections.

## Probabilistic & Generative Models

- **Linear Discriminant Analysis (LDA)**: Assumes normality, equal class covariances—maximizes separation.
- **Quadratic Discriminant Analysis (QDA)**: Like LDA but different class covariances.
- **Gaussian Mixture Discriminant Analysis**: Extension for non-linear boundaries.
- **Hidden Markov Model Classifier**: Probabilities over latent state sequences.

## Ensemble Methods (Beyond Above)

- **Bagging Classifier**: Aggregates multiple randomly created classifiers.
- **Stacking Classifier**: Combines predictions via meta-level model.
- **Voting Classifier**: Combines model outputs via majority or weighted vote.
- **Blending Ensemble**: Like stacking; uses a portion of data for blending.

## Rule-Based & Symbolic Learning

- **RIPPER**: Rule-learning with reduced error pruning.
- **CN2**: Extracts single rules at a time using entropy heuristic.
- **PART**: Builds partial decision trees to generate rules.
- **OneR**: Uses a single feature to create simple rules.
- **Decision Table**: Combines multiple rules into lookup structure.
- **FOIL**: Inductive logic programming-based rule learner.

## Evolutionary Algorithms

- **Genetic Programming for Classification**: Automatically evolves programs/rules to partition classes.
- **Learning Classifier System (LCS)**: Evolves a population of IF-THEN rules.
- **Genetic Algorithm Classifier**: Uses evolutionary search for optimal parameter settings.

## Prototype-Based and Clustering Methods (with classification adaptation)

- **Learning Vector Quantization (LVQ)**
- **Self-Organizing Maps (SOM) for Label Assignment**
- **k-means Class Assignment**: Assign clusters to classes post hoc.

## Specialty & Margin-Based Classifiers

- **Maximum Entropy Classifier (MaxEnt)**
- **Least Squares SVM (LSSVM)**
- **Twin SVM**
- **Fisher Linear Discriminant**
- **Nearest Centroid Classifier**
- **Mahalanobis Distance Classifier**
- **Sparse Representation Classifier**

## Cost-Sensitive and Imbalanced Data Algorithms

- **SMOTE-Bagged Classifier**: Uses SMOTE with bagging for imbalance.
- **Cost-Sensitive Trees/SVM/NN**: Modify loss functions to penalize misclassification.
- **CB-LDA**: Class-Balanced Linear Discriminant Analysis.

## Fuzzy and Hybrid Models

- **Fuzzy Decision Tree**
- **Fuzzy KNN**
- **Fuzzy Rule-Based Classifier**
- **Neuro-Fuzzy Classifier**

## Online/Learning with Limited Memory

- **Passive-Aggressive Classifier**: Online, updates model for each misclassified sample.
- **Online Perceptron**
- **Online SGD Classifier**: Uses stochastic gradient descent.

## Other Notable Specific Methods

- **Quadratic Programming Classifier**
- **Logistic Model Trees**
- **Oblique Decision Trees**
- **Multiclass ECOC Classifier (Error-Correcting Output Codes)**
- **One-vs-One, One-vs-Rest Wrappers**

## Highly Specialized and Domain-Specific Classifiers

For completeness, these would include:
- **Text Classifiers (e.g., n-gram, BERT-based)**
- **Graph Neural Network Classifier**
- **Relational/Inductive Logic Programming Classifier**
- **Event/Sequence Classifiers**
- **Ensemble of Nearest-Neighbor Models**
- **Multi-Instance Learning Classifier**
- **Multi-Label (Binary Relevance, Classifier Chains)**
- **Ordinal Classification (Ordinal Regression Models)**
- **Time Series/Sequence Discriminators (TSF, WEASEL, etc.)**

---

**Note:** 
- Citing over 200 individually *named* algorithms is challenging due to the combinatorial nature of many methods (e.g., SVM with dozens of different kernels, neural networks with thousands of architectures).
- Many software packages (like scikit-learn, XGBoost, LightGBM, etc.) further provide dozens of parameterized variants, which expands the practical list.

If you need **individual, one-line descriptions for 200+ unique algorithms** (including all exotic and research-variant names), or require a markdown-formatted exhaustive table, specify the desired granularity or request a focused sub-list from any category above. The above list is comprehensive for practical use and can be expanded further per your needs[1][2][3][4][5][10].

---

## Ai Algorithms Clustering

# AI_Algorithms_Clustering

A complete list of **200+ clustering algorithms** for AI and machine learning is not found in any single authoritative resource in the provided search results; most educational and industry sources list 10-20 major clustering algorithms, with some variants and hybrids. However, clusters in machine learning can be broadly categorized with well-known representatives, and collecting 200+ distinct clustering algorithms would require listing variants, adaptations, and domain-specific methods beyond mainstream categories[1][3][5][9][13].

**Below is a structured catalog of clustering algorithms (drawn from the major types and notable variants), with brief descriptions.** The most recognized approaches are presented first by major categories, with variants and hybrids expanding the count.

---

### Centroid-Based Clustering Algorithms

- **K-Means**: Assigns points to the nearest of K centroids, iteratively optimizing centroids to minimize variance within clusters[1][2][3][4][5].
- **Mini-Batch K-Means**: Runs K-means with small batches to scale for large data[1][9].
- **K-Medoids (PAM; Partitioning Around Medoids)**: Similar to K-means but uses medoids (actual data points) as centers, making it robust to outliers[3][9].
- **CLARA**: Extension of PAM using sampling for large data sets.
- **CLARANS**: Randomized search for clusters instead of checking all combinations.
- **K-Means++**: Improved initialization method for K-means for better convergence.

---

### Density-Based Clustering Algorithms

- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Groups points with sufficient close neighbors, labels low-density points as noise[1][3][4][5][7][9].
- **OPTICS (Ordering Points To Identify Clustering Structure)**: Extends DBSCAN to discover clusters of differing densities[1][3][9].
- **DENCLUE (Density-based Clustering by Linking spatial Clusters)**: Models clusters using density functions.
- **HDBSCAN (Hierarchical DBSCAN)**: Hierarchical extension of DBSCAN.

---

### Hierarchical Clustering Algorithms

- **Agglomerative Hierarchical Clustering**: "Bottom-up" approach; starts with each point as a cluster, merges closest clusters iteratively[1][3][4][5][9].
- **Divisive Hierarchical Clustering**: "Top-down" approach; starts with all points in one cluster then splits recursively.
- **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)**: Handles large datasets using hierarchical summaries, then clusters summaries[1][4][9].
- **CURE**: Uses multiple representative points for clusters.
- **CHAMELEON**: Relies on dynamic modeling for clusters.
- **ROCK**: Based on links instead of distance measures.

---

### Distribution-Based Clustering Algorithms

- **Gaussian Mixture Models (GMM)**: Fits clusters assuming underlying Gaussian distributions, assigning probabilistic memberships[1][3][4][5][9].
- **Expectation-Maximization Clustering (EM)**: Generalizes GMM for various distributions.
- **Bayesian Mixture Models**: Bayesian framework for mixture models.

---

### Spectral and Graph-Based Clustering Algorithms

- **Spectral Clustering**: Uses eigenvalues of similarity matrices to reduce dimensionality and find clusters[1][7][9].
- **Normalized Cuts**: Graph partitioning approach using eigenvectors.
- **Markov Clustering (MCL)**: Identifies clusters using random walks in graphs.
- **Modularity Maximization (Louvain, Girvan–Newman)**: Finds clusters in networks by optimizing modularity.
- **Clique Percolation Method**: Finds overlapping clusters in graphs.

---

### Fuzzy Clustering Algorithms

- **Fuzzy C-Means (FCM)**: Allows each data point to belong to multiple clusters with varying membership degrees[3].
- **Gustafson–Kessel Algorithm**: Variant for elliptical clusters.
- **Fuzzy K-Medoids**: Combination of fuzzy and medoid-based approaches.

---

### Grid-Based Clustering Algorithms

- **STING (Statistical Information Grid)**: Uses hierarchical grids for clustering.
- **CLIQUE**: Finds clusters in subspaces of high-dimensional data.
- **WaveCluster**: Uses wavelet transforms to find clusters.

---

### Model-Based and Hybrid Algorithms

- **Self-Organizing Maps (SOM, Kohonen maps)**: Neural networks for clustering and visualization.
- **Mixture of Probabilistic Principal Component Analyzers (MPPCA)**: Combines PCA and mixture models.
- **Autoencoder Clustering**: Deep learning models used for clustering representations.
- **Agglomerative Information Bottleneck**: Clustering using information theory.

---

### Constraint and Semi-Supervised Clustering Algorithms

- **COP-KMeans**: K-means with "must-link" and "cannot-link" constraints.
- **Seeded K-means**: Incorporates prior knowledge/seeds.

---

### Specialized, Robust, or Adaptive Clustering Algorithms

- **Mean Shift**: Finds clusters by seeking modes in density.
- **Affinity Propagation**: Exchanges messages to determine exemplars and clusters[1][9].
- **Ensemble Clustering (Consensus Clustering)**: Combines multiple clustering results.
- **Subspace Clustering**: Clusters in subspaces of high-dimensional data.
- **Time-Series Clustering (DTW-based, Shapelets)**: Specialized for temporal data.

---

### Domain-Specific Algorithms

- **Image Segmentation Algorithms**: Watershed, region growing, graph-cut clustering.
- **Text Clustering Algorithms**: Latent Dirichlet Allocation (LDA), Topic Modeling approaches.

---

### Notable KNN-Related Algorithms

- **KNN (K-Nearest Neighbors) Clustering**: While mainly for classification, can be adapted for clustering[2].

---

### Extension Approaches, Variants, and Meta-Algorithms

Beyond the above, hundreds of algorithms are described in the literature via:
- Hyperparameter variations (distance metrics, initialization schemes, linkage criteria)
- Algorithmic hybrids (combining clustering methods for novel applications)
- Domain constraints (clustering with structure, time, or spatial relationships)

---

### Expanding towards a 200+ list

To enumerate **200+ clustering algorithms** for academic review:
- List all major categories and all referenced algorithms and variants across peer-reviewed literature (for example, the scikit-learn documentation alone lists over a dozen main clustering methods and allows numerous parameterizations)[9].
- Include specialized algorithms from bioinformatics, market basket analysis, anomaly detection, multi-view clustering, consensus/ensemble clustering, and semi-supervised clustering methods.
- Include algorithmic variants by changing distance functions, initialization mechanisms, hierarchical linkage variants, or adapting models for specific data types[8][13].

---

**In summary:**  
AI and ML clustering algorithms number in the hundreds, encompassing main categories (centroid, density, hierarchical, distributional, fuzzy, grid, spectral, etc.), with each category branching into many specific methods, parameterizations, and adaptations for varied data and domains. Listing 200+ distinct algorithms entails accounting for mainstream, historic, domain-specific, and countless published variants and hybrids, as described in broad surveys and algorithm repositories[5][8][9][13].

If you need a complete enumerated table of 200+ algorithms with brief descriptions, this would require an extensive literature review and systematic expansion of the structured types and all published variants. If you want such a list, please specify the preferred detail depth or subfields (e.g., bioinformatics, text mining, graph clustering) to focus the expansion.

---

## Ai Algorithms Ensemble

# AI_Algorithms_Ensemble

I appreciate your interest, but I need to clarify that the search results do not contain a comprehensive list of 100+ ensemble methods and algorithms. While the search results mention several ensemble techniques and reference documents that claim to catalog 100+ AI algorithms, they don't actually provide the complete enumerated lists.

## What the Search Results Do Cover

The search results identify and describe several key ensemble methods:

**Popular Ensemble Techniques**

- **Random Forest**: Combines multiple decision trees using random sampling and feature selection, with final predictions based on majority voting (classification) or averaging (regression)[2][3]
- **Gradient Boosting**: Builds models sequentially where each new model corrects errors from previous ones[2]
  - **XGBoost** (Extreme Gradient Boosting): Advanced version with regularization for preventing overfitting[2]
  - **LightGBM** (Light Gradient Boosting Machine): Uses histogram-based approach for faster computation[2]
  - **CatBoost**: Designed specifically for categorical data with built-in encoding[2]
- **Bagging**: A foundational ensemble method[9]
- **AdaBoost**: Mentioned as an ensemble learning approach[2]
- **Stacking**: A ensemble method that combines multiple models[2]
- **Voting**: Combines predictions through voting mechanisms[1][3]
- **Averaging and Weighted Averaging**: Methods for aggregating regression predictions[5]

## Limitation

To obtain a complete list of 100+ ensemble methods, you would need to consult specialized academic papers, comprehensive machine learning textbooks, or dedicated algorithm repositories that provide exhaustive taxonomies of ensemble techniques. The search results reference such comprehensive guides but don't reproduce the full listings.

---

## Ai Algorithms Optimization

# AI_Algorithms_Optimization

Providing a comprehensive list of over 200 optimization algorithms for AI and ML with detailed descriptions is a significant undertaking. However, I can offer a broad overview of various optimization techniques used in machine learning, categorized by their types and applications. This will include some of the most commonly used algorithms along with brief descriptions.

## 1. **First-Order Optimization Algorithms**
These algorithms use gradient information to update parameters.

- **Gradient Descent (GD)**: Updates parameters in the direction of the negative gradient to minimize the loss function.
- **Stochastic Gradient Descent (SGD)**: Uses a single example from the training dataset to compute the gradient at each step.
- **Mini-Batch Gradient Descent**: Uses a small batch of examples to compute the gradient.
- **Momentum SGD**: Adds a fraction of the previous update to the current update to help escape local minima.
- **Nesterov Accelerated Gradient (NAG)**: Modifies momentum SGD by incorporating a "lookahead" step.
- **Adagrad**: Adapts the learning rate for each parameter based on past gradients.
- **RMSprop**: Similar to Adagrad but divides the learning rate by an exponentially decaying average of squared gradients.
- **Adam**: Combines Adagrad and RMSprop by adapting the learning rate for each parameter based on both the magnitude of recent gradient updates and the squared magnitude of recent updates.
- **AdamW**: A variant of Adam that decouples weight decay from the learning rate.

## 2. **Second-Order Optimization Algorithms**
These algorithms use both the first and second derivatives (Hessian matrix) to update parameters.

- **Newton's Method**: Uses the Hessian matrix to compute the update direction.
- **Quasi-Newton Methods**: Approximates the Hessian matrix using gradient information.
  - **BFGS (Broyden-Fletcher-Goldfarb-Shanno)**: A popular quasi-Newton method.
  - **L-BFGS (Limited-memory BFGS)**: A variant of BFGS that uses less memory.

## 3. **Metaheuristic Optimization Algorithms**
These algorithms are used for complex optimization problems and often involve heuristics.

- **Genetic Algorithm**: Inspired by natural selection and genetics, it uses crossover and mutation to search for optimal solutions.
- **Simulated Annealing**: Starts with a high temperature and gradually cools down to avoid local optima.
- **Ant Colony Optimization**: Inspired by the behavior of ants searching for food.
- **Particle Swarm Optimization**: Inspired by the social behavior of bird flocking or fish schooling.
- **Tabu Search**: Uses memory to avoid revisiting recently explored solutions.
- **Iterated Local Search (ILS)**: Combines local search with random perturbations to escape local optima.

## 4. **Bracketing Algorithms**
Used for single-variable optimization problems where the optimum is known to exist within a specific range.

- **Fibonacci Search**: Uses Fibonacci numbers to divide the search interval.
- **Golden Section Search**: Similar to Fibonacci search but uses the golden ratio.
- **Bisection Method**: Repeatedly divides the interval in half to find the root of a function.

## 5. **Local Descent Algorithms**
Used for multi-variable optimization problems.

- **Line Search**: Finds the optimal step size in a given direction.
- **Brent-Dekker Algorithm**: A robust line search algorithm.

## 6. **Derivative-Free Optimization Algorithms**
These algorithms do not require gradient information.

- **Grid Search**: Exhaustively searches through a predefined set of hyperparameters.
- **Random Search**: Randomly samples hyperparameters from a given distribution.
- **Bayesian Optimization**: Uses probabilistic models to efficiently search for optimal hyperparameters.
- **Evolution Strategies**: Uses evolutionary principles to optimize parameters without gradients.

## 7. **Specialized Optimization Algorithms**
- **Conjugate Gradient**: Used for solving systems of linear equations.
- **Trust Region Methods**: Limits the step size to ensure convergence.
- **Coordinate Descent**: Optimizes one variable at a time while keeping others fixed.

## 8. **Deep Learning Specific Optimization Algorithms**
- **Adadelta**: An adaptive learning rate method similar to RMSprop but with a different approach to handling the squared gradients.
- **Nadam**: Combines Adam with NAG for faster convergence.

## 9. **Other Optimization Techniques**
- **Gradient Clipping**: Prevents exploding gradients by limiting their magnitude.
- **Learning Rate Schedulers**: Adjusts the learning rate during training to improve convergence.

This list covers a wide range of optimization algorithms used in AI and ML, but it is not exhaustive. Each algorithm has its strengths and is suited for different types of problems and datasets. For a comprehensive list of over 200 algorithms, one would need to delve into more specialized literature and research papers in the field.

---

## Ai Algorithms Regression

# AI_Algorithms_Regression

It is not currently practical to list **200+ named regression algorithms** with descriptions from authoritative sources, as no single comprehensive, canonical list of that length exists in public, academic, or industry literature. Machine learning regression algorithms number in the dozens when counting standard, widely-used, and specialized algorithms—extending such a list to 200 typically requires including subtypes, variants, custom ensembles, or obsolete/rarely-used approaches. However, I can provide an extensive and categorized list of major regression algorithms, their common variants, families, and extensions, covering the breadth of what is documented in the literature and practice.

Below is a comprehensive list of **regression algorithms, families, and common variants**, along with concise descriptions. This list is organized by technique and modeling approach. For each broader family, major variations and specific models are included to approach maximum breadth, while maintaining clarity and completeness.

---

### Linear and Generalized Linear Regression Algorithms

- **Linear Regression**  
  Models a linear relationship between predictors and a continuous target variable[1][2][4].

- **Multiple Linear Regression**  
  Extension to multiple predictors, estimating coefficients per feature.

- **Polynomial Regression**  
  Fits a polynomial (non-linear) relationship by adding polynomial terms of the predictors[2][4].

- **Ridge Regression** (L2 Regularization)  
  Adds L2 penalty to reduce coefficients and handle multicollinearity[2][4].

- **Lasso Regression** (L1 Regularization)  
  Adds L1 penalty, promoting sparsity and feature selection[2][4].

- **Elastic Net Regression**  
  Combines L1 and L2 regularization[2][4].

- **Stepwise Regression**  
  Iteratively adds or removes predictors based on criteria (AIC/BIC/p-value)[2][3].

- **Orthogonal Regression**  
  Minimizes orthogonal distances, handling errors in both X and Y.

- **Total Least Squares Regression**  
  Variant focusing on small errors in both predictors and outcomes.

- **Quantile Regression**  
  Models specific conditional quantiles (e.g., median) of the response variable[2].

- **Robust Regression**  
  Techniques that reduce the influence of outliers (e.g., Huber, RANSAC).

- **Principal Component Regression (PCR)**  
  Performs PCA for dimensionality reduction, then regression[2].

- **Partial Least Squares Regression (PLSR)**  
  Projects predictors to maximize relevance to the outcome[2].

- **Weighted Least Squares Regression**  
  Assigns weights to data points for heteroscedastic errors.

- **Generalized Linear Models (GLMs):**
  - **Poisson Regression:** For count data[2].
  - **Binomial/Logistic Regression:** For probabilities/binary outcomes[2].
  - **Probit Regression:** Similar to logistic, uses normal CDF.
  - **Negative Binomial Regression:** For overdispersed count data.
  - **Gamma Regression:** For skewed continuous positive responses.
  - **Inverse Gaussian Regression:** For non-negative, skewed outcomes.

- **Bayesian Linear Regression**  
  Incorporates prior information and provides uncertainty estimates[2].

- **Least Angle Regression (LARS)**  
  Efficient for high-dimensional data; related to Lasso[3].

---

### Nonlinear Regression Algorithms

- **Nonlinear Least Squares Regression**  
  Fits explicitly specified nonlinear functions.

- **Spline Regression (Basis Expansion, B-splines, Natural Splines, Cubic Splines)**  
  Piecewise polynomial fits for complex curves.

- **Locally Estimated Scatterplot Smoothing (LOESS/LOWESS)**  
  Local polynomial regression fits across data[3].

- **Multivariate Adaptive Regression Splines (MARS)**  
  Flexible model using piecewise linear basis functions[3].

---

### Tree-Based and Ensemble Regression Algorithms

- **Decision Tree Regression**  
  Rule-based splits on features to minimize error[1][2][3][4].

- **Random Forest Regression**  
  Ensemble of decision trees; reduces variance and overfitting[1][2][4].

- **Extra Trees Regression (Extremely Randomized Trees)**  
  Like Random Forest but with more randomized splits.

- **Gradient Boosted Regression Trees (GBRT/GBM)**  
  Sequential, additive ensemble minimizing residuals.

- **XGBoost Regression**  
  Advanced, regularized gradient boosting tree algorithm.

- **LightGBM Regression**  
  Efficient gradient boosting with leaf-wise splits.

- **CatBoost Regression**  
  Categorical feature support; state-of-the-art boosting.

- **Stochastic Gradient Boosting**  
  Boosting with random sub-samples per iteration.

- **AdaBoost Regression**  
  Boosts weak learners using weighted data points.

- **Bagging Regression**  
  Aggregates multiple regressors trained on bootstrap samples.

- **Model Trees (e.g., M5, Cubist)**  
  Tree leaves contain linear regression models[3].

---

### Kernel and Support Vector Algorithms

- **Support Vector Regression (SVR)**  
  Uses kernel functions to model nonlinear relationships, margin-based loss[2][4].

- **Kernel Ridge Regression**  
  Ridge regression in kernel-transformed space.

- **Relevance Vector Regression**  
  Bayesian sparse kernel approach.

- **Gaussian Process Regression (GPR/Kriging)**  
  Nonparametric Bayesian regression with uncertainty estimates[4].

---

### Instance-based and Memory-Based Regression Algorithms

- **k-Nearest Neighbors Regression (kNN)**  
  Predicts target based on local neighbors' values[4].

- **Locally Weighted Regression**  
  Fits local models around each prediction point.

- **Learning Vector Quantization (LVQ) Regression**  
  Prototype-based method adapted for regression[3].

- **Self-Organizing Maps (SOM) Regression**  
  Neural-inspired, unsupervised mapping for regression[3].

---

### Neural Networks and Deep Learning for Regression

- **Multi-Layer Perceptron (MLP) Regression**  
  Feed-forward neural networks for nonlinear regression.

- **Deep Neural Network Regression**  
  Multiple hidden layers for complex pattern extraction.

- **Convolutional Neural Network Regression**  
  For image/sequence regression tasks.

- **Recurrent Neural Network Regression (RNN, LSTM, GRU)**  
  For sequential/temporal regression.

- **Residual Networks (ResNet) Regression**

- **Bayesian Neural Network Regression**  
  Neural nets with probabilistic weights.

---

### Probabilistic and Bayesian Regression Algorithms

- **Naive Bayes Regression**  
  Limited use due to independence assumption, but possible for some cases[3].

- **Bayesian Network Regression**  
  Probabilistic graphical model for predicting real values[3].

- **Averaged One-Dependence Estimators (AODE) Regression**  
  Class of Bayesian regressors.

- **Gaussian Mixture Regression**  
  Uses Gaussian mixtures for flexible response modeling.

---

### Specialized and Structured Regression

- **Quantile Random Forest Regression**  
  Extends random forests to predict conditional quantiles.

- **Isotonic Regression**  
  Fits monotonic (non-decreasing or non-increasing) functions.

- **Theil-Sen Regression**  
  Robust non-parametric method using median of slopes.

- **RANSAC Regression**  
  Robust to outliers via random sampling.

- **Huber Regression**  
  Loss function less sensitive to outliers than squared error.

- **LAD Regression (Least Absolute Deviations)**  
  Minimizes absolute error for robustness.

- **Tobit Regression**  
  For censored (truncated) outcome variables.

- **Survival/Cox Regression**  
  Models event times, handling censoring[2].

- **Time Series Regression Models**  
  Incorporates autocorrelation, AR/ARMA/ARIMA/ARIMAX methods[2].

- **Panel Data Regression (Fixed/Random Effects)**  
  Models longitudinal/multi-level data[2].

---

### Dimensionality and Manifold Methods

- **Manifold Regression (e.g., Locally Linear Embedding + Regression)**  
  Provides regression on nonlinear manifolds.

---

### Combinatory and Meta-Regression Techniques

- **Stacked Regression (Stacking/Blending/Ensembling)**  
  Aggregates output of multiple regressors.

- **Bayesian Model Averaging Regression**  
  Weighted ensemble based on posterior probabilities.

- **Greedy, Forward/Backward Feature Selection Regression**

- **Ensemble Regression using Voting/Mixture-of-Experts**

---

### Uncommon, Advanced, or Niche Algorithms

- **Zero-Inflated Models (Zero-Inflated Poisson/Gaussian)**  
  For outcomes with excess zeros.

- **Beta Regression**  
  For proportions or rates between 0 and 1.

- **Dirichlet Regression**  
  For compositional (proportion) data.

- **Log-Log/Logit Regression**  
  For special data scale/transformations.

- **Bayesian Additive Regression Trees (BART)**

- **Projection Pursuit Regression**

---

This catalog—when fully expanded with all named variants, historical methods, combinatory approaches, different neural architectures, regularization and feature engineering strategies, and data-specific regressors—approaches the breadth of **200 distinct regression methods** used in AI and ML practice and research.

**No single published list explicitly enumerates and describes 200 “regression algorithms” by unique names; instead, combinations, variants, and meta-approaches are used to achieve this diversity**[2][3][4].

If you require the list in an itemized form with very brief descriptions (as above), I can format it further into a numbered, flat list for you. Let me know your preference or if you wish to see the methods grouped in another way.

---

## Ml Libraries Python

# ML_Libraries_Python

You can find well over **300 Python libraries for machine learning** on curated GitHub lists, with each project linked to its repository for further exploration. The most efficient way to access 300+ libraries with GitHub URLs is via these curated directories, which are actively maintained and include descriptions and repository links:

**Primary Comprehensive Lists:**

- **Best-of Machine Learning with Python**
  - [https://github.com/lukasmasuch/best-of-ml-python](https://github.com/lukasmasuch/best-of-ml-python)
  - Contains **920** ranked open-source Python ML projects, each with GitHub URLs[5].
- **ML-Libraries-Python**
  - [https://github.com/Habtish-yimer/ML-Libraries-Python](https://github.com/Habtish-yimer/ML-Libraries-Python)
  - Mirrors the above, listing 920+ ML Python libraries with URLs[9].
- **Awesome Python Machine Learning**
  - [https://github.com/sorend/awesome-python-machine-learning](https://github.com/sorend/awesome-python-machine-learning)
  - Curated list of *active* Python ML frameworks, tools, and resources, all with repository links[7].
- **Awesome Machine Learning**
  - [https://github.com/josephmisiti/awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning)
  - Well-known, frequently updated ML library aggregator with direct GitHub links[13].

**Examples of Major Libraries with GitHub URLs:**
- **scikit-learn:** [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
- **PyTorch:** [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- **TensorFlow:** [https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
- **Keras:** [https://github.com/keras-team/keras](https://github.com/keras-team/keras)
- **XGBoost:** [https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)
- **LightGBM:** [https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
- **CatBoost:** [https://github.com/catboost/catboost](https://github.com/catboost/catboost)
- **PyCaret:** [https://github.com/pycaret/pycaret](https://github.com/pycaret/pycaret)
- **H2O.ai:** [https://github.com/h2oai/h2o-3](https://github.com/h2oai/h2o-3)
- **Eli5:** [https://github.com/TeamHG-Memex/eli5](https://github.com/TeamHG-Memex/eli5)
- **PyBrain:** [https://github.com/pybrain/pybrain](https://github.com/pybrain/pybrain)
- **Dist-Keras:** [https://github.com/JoeriHermans/dist-keras](https://github.com/JoeriHermans/dist-keras)
- **Theano:** [https://github.com/Theano/Theano](https://github.com/Theano/Theano)

**Instructions for Bulk Access:**
- Visit any of the comprehensive GitHub repository lists above.
- The repositories are categorized (e.g., deep learning, NLP, AutoML).
- Each entry provides the **name, description, and direct GitHub link**.
- You can filter or search by tags or domain-specific needs.

**Advantages of Using These Lists:**
- They allow instant navigation to the GitHub project, enabling cloning, installation, or review.
- They contain more than the requested 300 libraries, with continuous updates and rankings for relevance/popularity.

If you need the list in a **machine-readable format** (e.g., CSV or plain TXT), most of these repositories include the library name and URL in Markdown, which can be easily parsed or exported. 

For your requirement of 300+ Python machine learning libraries—**visit** and utilize these curated GitHub lists for direct URLs, descriptions, and further exploration[2][5][7][9][13].

---

# PART 2: LATEST AI MODELS BY COMPANY

**Total Models:** 196

## Alibaba (Qwen)

**Total:** 39 models

| Model | URL |
|-------|-----|
| Qwen3 235B | https://huggingface.co/Qwen/Qwen3-235B |
| Qwen3 235B (Reasoning) | https://huggingface.co/Qwen |
| Qwen3 32B | https://huggingface.co/Qwen/Qwen3-32B |
| Qwen3 32B (Reasoning) | https://huggingface.co/Qwen |
| Qwen3 30B A3B | https://huggingface.co/Qwen |
| Qwen3 30B A3B (Reasoning) | https://huggingface.co/Qwen |
| Qwen3 14B | https://huggingface.co/Qwen/Qwen3-14B |
| Qwen3 14B (Reasoning) | https://huggingface.co/Qwen |
| Qwen3 8B | https://huggingface.co/Qwen/Qwen3-8B |
| Qwen3 8B (Reasoning) | https://huggingface.co/Qwen |
| Qwen3 4B | https://huggingface.co/Qwen/Qwen3-4B |
| Qwen3 4B (Reasoning) | https://huggingface.co/Qwen |
| Qwen3 1.7B | https://huggingface.co/Qwen/Qwen3-1.7B |
| Qwen3 1.7B (Reasoning) | https://huggingface.co/Qwen |
| Qwen3 0.6B | https://huggingface.co/Qwen/Qwen3-0.6B |
| Qwen3 0.6B (Reasoning) | https://huggingface.co/Qwen |
| Qwen2.5 Max | https://huggingface.co/Qwen/Qwen2.5-Max |
| Qwen2.5 Plus | https://huggingface.co/Qwen |
| Qwen2.5 Turbo | https://huggingface.co/Qwen |
| Qwen2.5 72B | https://huggingface.co/Qwen/Qwen2.5-72B |
| Qwen2.5 Instruct 72B | https://huggingface.co/Qwen/Qwen2.5-72B-Instruct |
| Qwen2.5 Instruct 32B | https://huggingface.co/Qwen/Qwen2.5-32B-Instruct |
| Qwen2.5 Instruct 14B | https://huggingface.co/Qwen/Qwen2.5-14B-Instruct |
| Qwen2.5 Instruct 7B | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
| Qwen2.5 Coder 32B | https://huggingface.co/Qwen/Qwen2.5-Coder-32B |
| Qwen2.5 Coder 7B | https://huggingface.co/Qwen/Qwen2.5-Coder-7B |
| Qwen2.5 VL 72B | https://huggingface.co/Qwen/Qwen2-VL-72B |
| Qwen2.5 VL 7B | https://huggingface.co/Qwen/Qwen2-VL-7B |
| Qwen2.5 Omni 7B | https://huggingface.co/Qwen |
| Qwen2 72B | https://huggingface.co/Qwen/Qwen2-72B |
| Qwen2 Instruct 7B | https://huggingface.co/Qwen/Qwen2-7B-Instruct |
| Qwen2-VL 72B | https://huggingface.co/Qwen/Qwen2-VL-72B |
| Qwen1.5 Chat 110B | https://huggingface.co/Qwen/Qwen1.5-110B-Chat |
| Qwen1.5 Chat 72B | https://huggingface.co/Qwen/Qwen1.5-72B-Chat |
| Qwen1.5 Chat 32B | https://huggingface.co/Qwen/Qwen1.5-32B-Chat |
| Qwen1.5 Chat 14B | https://huggingface.co/Qwen/Qwen1.5-14B-Chat |
| Qwen1.5 Chat 7B | https://huggingface.co/Qwen/Qwen1.5-7B-Chat |
| QwQ-32B | https://huggingface.co/Qwen/QwQ-32B |
| QwQ 32B-Preview | https://huggingface.co/Qwen/QwQ-32B-Preview" |

---

## Anthropic Claude

**Total:** 16 models

| Model | URL |
|-------|-----|
| Claude 4.5 Sonnet | https://www.anthropic.com/claude |
| Claude 4.1 Opus | https://www.anthropic.com/claude |
| Claude 4 Opus | https://www.anthropic.com/claude |
| Claude 4 Opus Thinking | https://www.anthropic.com/claude |
| Claude 4 Sonnet | https://www.anthropic.com/claude |
| Claude 4 Sonnet Thinking | https://www.anthropic.com/claude |
| Claude 3.7 Sonnet | https://www.anthropic.com/claude |
| Claude 3.7 Sonnet Thinking | https://www.anthropic.com/claude |
| Claude 3.5 Sonnet | https://www.anthropic.com/claude |
| Claude 3.5 Haiku | https://www.anthropic.com/claude |
| Claude 3 Opus | https://www.anthropic.com/claude |
| Claude 3 Sonnet | https://www.anthropic.com/claude |
| Claude 3 Haiku | https://www.anthropic.com/claude |
| Claude 2.1 | https://www.anthropic.com/claude |
| Claude 2.0 | https://www.anthropic.com/claude |
| Claude Instant | https://www.anthropic.com/claude |

---

## DeepSeek

**Total:** 18 models

| Model | URL |
|-------|-----|
| DeepSeek-V3.2-Exp | https://huggingface.co/deepseek-ai |
| DeepSeek-V3.1 | https://huggingface.co/deepseek-ai/DeepSeek-V3.1 |
| DeepSeek-V3 | https://huggingface.co/deepseek-ai/DeepSeek-V3 |
| DeepSeek-V2.5 | https://huggingface.co/deepseek-ai/DeepSeek-V2.5 |
| DeepSeek-V2 | https://huggingface.co/deepseek-ai/DeepSeek-V2 |
| DeepSeek-R1 | https://huggingface.co/deepseek-ai/DeepSeek-R1 |
| DeepSeek-R1 0528 | https://huggingface.co/deepseek-ai |
| DeepSeek R1 Distill Llama 70B | https://huggingface.co/deepseek-ai |
| DeepSeek R1 Distill Llama 8B | https://huggingface.co/deepseek-ai |
| DeepSeek R1 Distill Qwen 32B | https://huggingface.co/deepseek-ai |
| DeepSeek R1 Distill Qwen 14B | https://huggingface.co/deepseek-ai |
| DeepSeek R1 Distill Qwen 1.5B | https://huggingface.co/deepseek-ai |
| DeepSeek-Coder V2 | https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2 |
| DeepSeek-Coder V2 Lite | https://huggingface.co/deepseek-ai |
| DeepSeek-Coder | https://huggingface.co/deepseek-ai/deepseek-coder-33b-base |
| DeepSeek-VL2 | https://huggingface.co/deepseek-ai |
| DeepSeek Prover V2 671B | https://huggingface.co/deepseek-ai |
| Janus Pro 7B | https://huggingface.co/deepseek-ai |

---

## Google / DeepMind

**Total:** 35 models

| Model | URL |
|-------|-----|
| Gemini 2.5 Pro | https://deepmind.google/technologies/gemini/ |
| Gemini 2.5 Flash | https://deepmind.google/technologies/gemini/ |
| Gemini 2.5 Flash (Reasoning) | https://deepmind.google/technologies/gemini/ |
| Gemini 2.5 Flash-Lite | https://deepmind.google/technologies/gemini/ |
| Gemini 2.5 Flash-Lite (Reasoning) | https://deepmind.google/technologies/gemini/ |
| Gemini 2.0 Flash | https://deepmind.google/technologies/gemini/ |
| Gemini 2.0 Flash-Lite | https://deepmind.google/technologies/gemini/ |
| Gemini 2.0 Pro Experimental | https://deepmind.google/technologies/gemini/ |
| Gemini 1.5 Pro | https://deepmind.google/technologies/gemini/ |
| Gemini 1.5 Flash | https://deepmind.google/technologies/gemini/ |
| Gemini 1.5 Flash-8B | https://deepmind.google/technologies/gemini/ |
| Gemini 1.0 Pro | https://deepmind.google/technologies/gemini/ |
| Gemini 1.0 Ultra | https://deepmind.google/technologies/gemini/ |
| Gemini Ultra | https://deepmind.google/technologies/gemini/ |
| Gemma 3 27B | https://huggingface.co/google/gemma-3-27b |
| Gemma 3 12B | https://huggingface.co/google/gemma-3-12b |
| Gemma 3 4B | https://huggingface.co/google/gemma-3-4b |
| Gemma 3 1B | https://huggingface.co/google/gemma-3-1b |
| Gemma 3n E4B | https://huggingface.co/google |
| Gemma 2 27B | https://huggingface.co/google/gemma-2-27b |
| Gemma 2 9B | https://huggingface.co/google/gemma-2-9b |
| Gemma 7B | https://huggingface.co/google/gemma-7b |
| PaLM 2 | https://ai.google/discover/palm2/ |
| PaLM (Pathways Language Model) | https://ai.google/ |
| BERT | https://github.com/google-research/bert |
| T5 | https://github.com/google-research/text-to-text-transfer-transformer |
| LaMDA | https://blog.google/technology/ai/lamda/ |
| Imagen | https://imagen.research.google/ |
| Project Astra | https://deepmind.google/ |
| Project Mariner | https://deepmind.google/ |
| Jules | https://deepmind.google/ |
| SIMA | https://deepmind.google/ |
| Gopher | https://www.deepmind.com/ |
| Chinchilla | https://www.deepmind.com/ |
| AlphaFold | https://github.com/deepmind/alphafold |

---

## Meta (Llama)

**Total:** 22 models

| Model | URL |
|-------|-----|
| Llama 4 Scout | https://huggingface.co/meta-llama |
| Llama 4 Maverick | https://huggingface.co/meta-llama |
| Llama 4 Behemoth | https://huggingface.co/meta-llama |
| Llama 3.3 70B | https://huggingface.co/meta-llama/Llama-3.3-70B |
| Llama 3.2 90B Vision | https://huggingface.co/meta-llama/Llama-3.2-90B-Vision |
| Llama 3.2 11B Vision | https://huggingface.co/meta-llama/Llama-3.2-11B-Vision |
| Llama 3.2 3B | https://huggingface.co/meta-llama/Llama-3.2-3B |
| Llama 3.2 1B | https://huggingface.co/meta-llama/Llama-3.2-1B |
| Llama 3.1 405B | https://huggingface.co/meta-llama/Meta-Llama-3.1-405B |
| Llama 3.1 70B | https://huggingface.co/meta-llama/Meta-Llama-3.1-70B |
| Llama 3.1 8B | https://huggingface.co/meta-llama/Meta-Llama-3.1-8B |
| Llama 3 70B | https://huggingface.co/meta-llama/Meta-Llama-3-70B |
| Llama 3 8B | https://huggingface.co/meta-llama/Meta-Llama-3-8B |
| Llama 2 70B | https://huggingface.co/meta-llama/Llama-2-70b |
| Llama 2 13B | https://huggingface.co/meta-llama/Llama-2-13b |
| Llama 2 7B | https://huggingface.co/meta-llama/Llama-2-7b |
| Llama 65B | https://huggingface.co/meta-llama |
| Code Llama 70B | https://huggingface.co/codellama/CodeLlama-70b-hf |
| Chameleon | https://huggingface.co/facebook/chameleon |
| OPT 175B | https://huggingface.co/facebook/opt-175b |
| Galactica | https://huggingface.co/facebook/galactica-120b |
| Emu | https://ai.meta.com/ |

---

## Mistral AI

**Total:** 22 models

| Model | URL |
|-------|-----|
| Mistral Large 2 | https://huggingface.co/mistralai/Mistral-Large-2 |
| Mistral Large | https://huggingface.co/mistralai/Mistral-Large |
| Mistral Saba | https://huggingface.co/mistralai |
| Mistral Medium 3 | https://huggingface.co/mistralai |
| Mistral Medium | https://huggingface.co/mistralai |
| Mistral Small 3.2 | https://huggingface.co/mistralai |
| Mistral Small 3.1 | https://huggingface.co/mistralai |
| Mistral Small 3 | https://huggingface.co/mistralai |
| Mistral Small | https://huggingface.co/mistralai/Mistral-Small |
| Mistral NeMo | https://huggingface.co/mistralai/Mistral-Nemo |
| Mistral 7B | https://huggingface.co/mistralai/Mistral-7B-v0.1 |
| Mixtral 8x22B | https://huggingface.co/mistralai/Mixtral-8x22B |
| Mixtral 8x7B | https://huggingface.co/mistralai/Mixtral-8x7B-v0.1 |
| Ministral 8B | https://huggingface.co/mistralai/Ministral-8B |
| Ministral 3B | https://huggingface.co/mistralai/Ministral-3B |
| Magistral Medium | https://huggingface.co/mistralai |
| Magistral Small | https://huggingface.co/mistralai |
| Codestral | https://huggingface.co/mistralai/Codestral-22B-v0.1 |
| Codestral-Mamba | https://huggingface.co/mistralai/Codestral-Mamba |
| Devstral | https://huggingface.co/mistralai |
| Pixtral Large | https://huggingface.co/mistralai/Pixtral-Large |
| Pixtral 12B | https://huggingface.co/mistralai/Pixtral-12B |

---

## OpenAI

**Total:** 35 models

| Model | URL |
|-------|-----|
| GPT-5 | https://openai.com/ |
| GPT-5 mini | https://openai.com/ |
| GPT-5 nano | https://openai.com/ |
| GPT-4.5 | https://openai.com/ |
| GPT-4.1 | https://openai.com/ |
| GPT-4.1 mini | https://openai.com/ |
| GPT-4.1 nano | https://openai.com/ |
| GPT-4o | https://platform.openai.com/docs/models/gpt-4o |
| GPT-4o mini | https://platform.openai.com/docs/models/gpt-4o-mini |
| GPT-4o Realtime | https://platform.openai.com/docs/models |
| GPT-4o Audio | https://platform.openai.com/docs/models |
| GPT-4 Turbo | https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4 |
| GPT-4 Vision | https://platform.openai.com/docs/models |
| GPT-4 | https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4 |
| GPT-3.5 Turbo | https://platform.openai.com/docs/models/gpt-3-5-turbo |
| GPT-3.5 Turbo Instruct | https://platform.openai.com/docs/models |
| GPT-OSS-120B | https://openai.com/open-models/ |
| GPT-OSS-20B | https://openai.com/open-models/ |
| o1 | https://openai.com/o1/ |
| o1-mini | https://openai.com/o1/ |
| o1-preview | https://openai.com/o1/ |
| o1-pro | https://openai.com/o1/ |
| o3 | https://openai.com/o3/ |
| o3-mini | https://openai.com/o3/ |
| o3-pro | https://openai.com/o3/ |
| o4-mini (high) | https://openai.com/ |
| Codex | https://openai.com/ |
| Whisper | https://github.com/openai/whisper |
| Whisper Large v3 Turbo | https://huggingface.co/openai/whisper-large-v3-turbo |
| DALLÂ·E 3 | https://openai.com/dall-e-3 |
| DALLÂ·E 2 | https://openai.com/dall-e-2 |
| Sora | https://openai.com/sora |
| GPT-1 | https://openai.com/ |
| GPT-2 | https://github.com/openai/gpt-2 |
| GPT-3 | https://openai.com/ |

---

## xAI (Grok)

**Total:** 9 models

| Model | URL |
|-------|-----|
| Grok 4 | https://x.ai/ |
| Grok 3 | https://x.ai/ |
| Grok 3 Reasoning Beta | https://x.ai/ |
| Grok 3 mini | https://x.ai/ |
| Grok 3 mini Reasoning (low) | https://x.ai/ |
| Grok 3 mini Reasoning (high) | https://x.ai/ |
| Grok 2 | https://x.ai/ |
| Grok-1 | https://github.com/xai-org/grok-1 |
| Grok Beta | https://x.ai/ |

---

## 📊 COLLECTION SUMMARY

### Part 1: AI Algorithms & Libraries

- **Ai Algorithms Classification**
- **Ai Algorithms Clustering**
- **Ai Algorithms Ensemble**
- **Ai Algorithms Optimization**
- **Ai Algorithms Regression**
- **Ml Libraries Python**

**Total Categories:** 6

### Part 2: Latest AI Models

- **Alibaba (Qwen):** 39 models
- **Anthropic Claude:** 16 models
- **DeepSeek:** 18 models
- **Google / DeepMind:** 35 models
- **Meta (Llama):** 22 models
- **Mistral AI:** 22 models
- **OpenAI:** 35 models
- **xAI (Grok):** 9 models

**Total Models:** 196
**Total Companies:** 8

### Estimated Total Resources

- **Algorithms & Libraries:** ~900
- **Latest Models:** 196
- **TOTAL NEW:** ~1,096

---

**Created:** October 29, 2025
**Source:** Epic search using Perplexity API + User provided models
