# 🌐 PLATFORMS CONSOLIDATED COLLECTION

## Recursos Consolidados de Todas as Principais Plataformas

**Total de Recursos:** 2,229+
**Plataformas:** 5

---

## 📚 TABLE OF CONTENTS

1. [Awesome Machine Learning](#awesome-machine-learning) - 1,268 recursos
2. [Transferlearning](#transferlearning) - 151 recursos
3. [Awesome Deep Learning](#awesome-deep-learning) - 610 recursos
4. [HuggingFace Top Models](#huggingface-top-models) - 100 modelos
5. [HuggingFace Top Datasets](#huggingface-top-datasets) - 100 datasets

---

# AWESOME MACHINE LEARNING

**Source:** https://github.com/josephmisiti/awesome-machine-learning
**Stars:** 65,000+
**Resources:** 1,268+

## Content

# Awesome Machine Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![Track Awesome List](https://www.trackawesomelist.com/badge.svg)](https://www.trackawesomelist.com/josephmisiti/awesome-machine-learning/)

A curated list of awesome machine learning frameworks, libraries and software (by language). Inspired by `awesome-php`.

_If you want to contribute to this list (please do), send me a pull request or contact me [@josephmisiti](https://twitter.com/josephmisiti)._
Also, a listed repository should be deprecated if:

* Repository's owner explicitly says that "this library is not maintained".
* Not committed for a long time (2~3 years).

Further resources:

* For a list of free machine learning books available for download, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md).

* For a list of professional machine learning events, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/events.md).

* For a list of (mostly) free machine learning courses available online, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/courses.md).

* For a list of blogs and newsletters on data science and machine learning, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/blogs.md).

* For a list of free-to-attend meetups and local events, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/meetups.md).

## Table of Contents

### Frameworks and Libraries
<!-- MarkdownTOC depth=4 -->
<!-- Contents-->
- [Awesome Machine Learning ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](#awesome-machine-learning-)
  - [Table of Contents](#table-of-contents)
    - [Frameworks and Libraries](#frameworks-and-libraries)
    - [Tools](#tools)
  - [APL](#apl)
      - [General-Purpose Machine Learning](#apl-general-purpose-machine-learning)
  - [C](#c)
      - [General-Purpose Machine Learning](#c-general-purpose-machine-learning)
      - [Computer Vision](#c-computer-vision)
  - [C++](#cpp)
      - [Computer Vision](#cpp-computer-vision)
      - [General-Purpose Machine Learning](#cpp-general-purpose-machine-learning)
      - [Natural Language Processing](#cpp-natural-language-processing)
      - [Speech Recognition](#cpp-speech-recognition)
      - [Sequence Analysis](#cpp-sequence-analysis)
      - [Gesture Detection](#cpp-gesture-detection)
      - [Reinforcement Learning](#cpp-reinforcement-learning)
  - [Common Lisp](#common-lisp)
      - [General-Purpose Machine Learning](#common-lisp-general-purpose-machine-learning)
  - [Clojure](#clojure)
      - [Natural Language Processing](#clojure-natural-language-processing)
      - [General-Purpose Machine Learning](#clojure-general-purpose-machine-learning)
      - [Deep Learning](#clojure-deep-learning)
      - [Data Analysis](#clojure-data-analysis--data-visualization)
      - [Data Visualization](#clojure-data-visualization)
      - [Interop](#clojure-interop)
      - [Misc](#clojure-misc)
      - [Extra](#clojure-extra)
  - [Crystal](#crystal)
      - [General-Purpose Machine Learning](#crystal-general-purpose-machine-learning)
  - [Elixir](#elixir)
      - [General-Purpose Machine Learning](#elixir-general-purpose-machine-learning)
      - [Natural Language Processing](#elixir-natural-language-processing)
  - [Erlang](#erlang)
      - [General-Purpose Machine Learning](#erlang-general-purpose-machine-learning)
  - [Fortran](#fortran)
      - [General-Purpose Machine Learning](#fortran-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#fortran-data-analysis--data-visualization)
  - [Go](#go)
      - [Natural Language Processing](#go-natural-language-processing)
      - [General-Purpose Machine Learning](#go-general-purpose-machine-learning)
      - [Spatial analysis and geometry](#go-spatial-analysis-and-geometry)
      - [Data Analysis / Data Visualization](#go-data-analysis--data-visualization)
      - [Computer vision](#go-computer-vision)
      - [Reinforcement learning](#go-reinforcement-learning)
  - [Haskell](#haskell)
      - [General-Purpose Machine Learning](#haskell-general-purpose-machine-learning)
  - [Java](#java)
      - [Natural Language Processing](#java-natural-language-processing)
      - [General-Purpose Machine Learning](#java-general-purpose-machine-learning)
      - [Speech Recognition](#java-speech-recognition)
      - [Data Analysis / Data Visualization](#java-data-analysis--data-visualization)
      - [Deep Learning](#java-deep-learning)
  - [Javascript](#javascript)
      - [Natural Language Processing](#javascript-natural-language-processing)
      - [Data Analysis / Data Visualization](#javascript-data-analysis--data-visualization)
      - [General-Purpose Machine Learning](#javascript-general-purpose-machine-learning)
      - [Misc](#javascript-misc)
      - [Demos and Scripts](#javascript-demos-and-scripts)
  - [Julia](#julia)
      - [General-Purpose Machine Learning](#julia-general-purpose-machine-learning)
      - [Natural Language Processing](#julia-natural-language-processing)
      - [Data Analysis / Data Visualization](#julia-data-analysis--data-visualization)
      - [Misc Stuff / Presentations](#julia-misc-stuff--presentations)
  - [Kotlin](#kotlin)
      - [Deep Learning](#kotlin-deep-learning)
  - [Lua](#lua)
      - [General-Purpose Machine Learning](#lua-general-purpose-machine-learning)
      - [Demos and Scripts](#lua-demos-and-scripts)
  - [Matlab](#matlab)
      - [Computer Vision](#matlab-computer-vision)
      - [Natural Language Processing](#matlab-natural-language-processing)
      - [General-Purpose Machine Learning](#matlab-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#matlab-data-analysis--data-visualization)
  - [.NET](#net)
      - [Computer Vision](#net-computer-vision)
      - [Natural Language Processing](#net-natural-language-processing)
      - [General-Purpose Machine Learning](#net-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#net-data-analysis--data-visualization)
  - [Objective C](#objective-c)
    - [General-Purpose Machine Learning](#objective-c-general-purpose-machine-learning)
  - [OCaml](#ocaml)
    - [General-Purpose Machine Learning](#ocaml-general-purpose-machine-learning)
  - [OpenCV](#opencv)
    - [Computer Vision](#opencv-Computer-Vision)
    - [Text-Detection](#Text-Character-Number-Detection)
  - [Perl](#perl)
    - [Data Analysis / Data Visualization](#perl-data-analysis--data-visualization)
    - [General-Purpose Machine Learning](#perl-general-purpose-machine-learning)
  - [Perl 6](#perl-6)
    - [Data Analysis / Data Visualization](#perl-6-data-analysis--data-visualization)
    - [General-Purpose Machine Learning](#perl-6-general-purpose-machine-learning)
  - [PHP](#php)
    - [Natural Language Processing](#php-natural-language-processing)
    - [General-Purpose Machine Learning](#php-general-purpose-machine-learning)
  - [Python](#python)
      - [Computer Vision](#python-computer-vision)
      - [Natural Language Processing](#python-natural-language-processing)
      - [General-Purpose Machine Learning](#python-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#python-data-analysis--data-visualization)
      - [Misc Scripts / iPython Notebooks / Codebases](#python-misc-scripts--ipython-notebooks--codebases)
      - [Neural Networks](#python-neural-networks)
      - [Survival Analysis](#python-survival-analysis)
      - [Federated Learning](#python-federated-learning)
      - [Kaggle Competition Source Code](#python-kaggle-competition-source-code)
      - [Reinforcement Learning](#python-reinforcement-learning)
      - [Speech Recognition](#python-speech-recognition)
  - [Ruby](#ruby)
      - [Natural Language Processing](#ruby-natural-language-processing)
      - [General-Purpose Machine Learning](#ruby-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#ruby-data-analysis--data-visualization)
      - [Misc](#ruby-misc)
  - [Rust](#rust)
      - [General-Purpose Machine Learning](#rust-general-purpose-machine-learning)
      - [Deep Learning](#rust-deep-learning)
      - [Natural Language Processing](#rust-natural-language-processing)
  - [R](#r)
      - [General-Purpose Machine Learning](#r-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#r-data-analysis--data-visualization)
  - [SAS](#sas)
      - [General-Purpose Machine Learning](#sas-general-purpose-machine-learning)
      - [Data Analysis / Data Visualization](#sas-data-analysis--data-visualization)
      - [Natural Language Processing](#sas-natural-language-processing)
      - [Demos and Scripts](#sas-demos-and-scripts)
  - [Scala](#scala)
      - [Natural Language Processing](#scala-natural-language-processing)
      - [Data Analysis / Data Visualization](#scala-data-analysis--data-visualization)
      - [General-Purpose Machine Learning](#scala-general-purpose-machine-learning)
  - [Scheme](#scheme)
      - [Neural Networks](#scheme-neural-networks)
  - [Swift](#swift)
      - [General-Purpose Machine Learning](#swift-general-purpose-machine-learning)
  - [TensorFlow](#tensorflow)
      - [General-Purpose Machine Learning](#tensorflow-general-purpose-machine-learning)

### [Tools](#tools-1)

- [Neural Networks](#tools-neural-networks)
- [Misc](#tools-misc)


[Credits](#credits)

<!-- /MarkdownTOC -->

<a name="apl"></a>
## APL

<a name="apl-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning
* [naive-apl](https://github.com/mattcunningham/naive-apl) - Naive Bayesian Classifier implementation in APL. **[Deprecated]**

<a name="c"></a>
## C

<a name="c-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning
* [Darknet](https://github.com/pjreddie/darknet) - Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.
* [Recommender](https://github.com/GHamrouni/Recommender) - A C library for product recommendations/suggestions using collaborative filtering (CF).
* [Hybrid Recommender System](https://github.com/SeniorSA/hybrid-rs-trainner) - A hybrid recommender system based upon scikit-learn algorithms. **[Deprecated]**
* [neonrvm](https://github.com/siavashserver/neonrvm) - neonrvm is an open source machine learning library based on RVM technique. It's written in C programming language and comes with Python programming language bindings.
* [cONNXr](https://github.com/alrevuelta/cONNXr) - An `ONNX` runtime written in pure C (99) with zero dependencies focused on small embedded devices. Run inference on your machine learning models no matter which framework you train it with. Easy to install and compiles everywhere, even in very old devices.
* [libonnx](https://github.com/xboot/libonnx) - A lightweight, portable pure C99 onnx inference engine for embedded devices with hardware acceleration support.
* [onnx-c](https://github.com/onnx/onnx-c) - A lightweight C library for ONNX model inference, optimized for performance and portability across platforms.

<a name="c-computer-vision"></a>
#### Computer Vision

* [CCV](https://github.com/liuliu/ccv) - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library.
* [VLFeat](http://www.vlfeat.org/) - VLFeat is an open and portable library of computer vision algorithms, which has a Matlab toolbox.
* [YOLOv8](https://github.com/ultralytics/ultralytics) - Ultralytics' YOLOv8 implementation with C++ support for real-time object detection and tracking, optimized for edge devices.

<a name="cpp"></a>
## C++

<a name="cpp-computer-vision"></a>
#### Computer Vision

* [DLib](http://dlib.net/imaging.html) - DLib has C++ and Python interfaces for face detection and training general object detectors.
* [EBLearn](http://eblearn.sourceforge.net/) - Eblearn is an object-oriented C++ library that implements various machine learning models **[Deprecated]**
* [OpenCV](https://opencv.org) - OpenCV has C++, C, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS.
* [VIGRA](https://github.com/ukoethe/vigra) - VIGRA is a genertic cross-platform C++ computer vision and machine learning library for volumes of arbitrary dimensionality with Python bindings.
* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) - A real-time multi-person keypoint detection library for body, face, hands, and foot estimation

<a name="cpp-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Speedster](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/speedster) -Automatically apply SOTA optimization techniques to achieve the maximum inference speed-up on your hardware. [DEEP LEARNING]
* [BanditLib](https://github.com/jkomiyama/banditlib) - A simple Multi-armed Bandit library. **[Deprecated]**
* [Caffe](https://github.com/BVLC/caffe) - A deep learning framework developed with cleanliness, readability, and speed in mind. [DEEP LEARNING]
* [CatBoost](https://github.com/catboost/catboost) - General purpose gradient boosting on decision trees library with categorical features support out of the box. It is easy to install, contains fast inference implementation and supports CPU and GPU (even multi-GPU) computation.
* [CNTK](https://github.com/Microsoft/CNTK) - The Computational Network Toolkit (CNTK) by Microsoft Research, is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph.
* [CUDA](https://code.google.com/p/cuda-convnet/) - This is a fast C++/CUDA implementation of convolutional [DEEP LEARNING]
* [DeepDetect](https://github.com/jolibrain/deepdetect) - A machine learning API and server written in C++11. It makes state of the art machine learning easy to work with and integrate into existing applications.
* [Distributed Machine learning Tool Kit (DMTK)](http://www.dmtk.io/) - A distributed machine learning (parameter server) framework by Microsoft. Enables training models on large data sets across multiple machines. Current tools bundled with it include: LightLDA and Distributed (Multisense) Word Embedding.
* [DLib](http://dlib.net/ml.html) - A suite of ML tools designed to be easy to imbed in other applications.
* [DSSTNE](https://github.com/amznlabs/amazon-dsstne) - A software library created by Amazon for training and deploying deep neural networks using GPUs which emphasizes speed and scale over experimental flexibility.
* [DyNet](https://github.com/clab/dynet) - A dynamic neural network library working well with networks that have dynamic structures that change for every training instance. Written in C++ with bindings in Python.
* [Fido](https://github.com/FidoProject/Fido) - A highly-modular C++ machine learning library for embedded electronics and robotics.
* [FlexML](https://github.com/ozguraslank/flexml) - Easy-to-use and flexible AutoML library for Python.
* [igraph](http://igraph.org/) - General purpose graph library.
* [Intel® oneAPI Data Analytics Library](https://github.com/oneapi-src/oneDAL) - A high performance software library developed by Intel and optimized for Intel's architectures. Library provides algorithmic building blocks for all stages of data analytics and allows to process data in batch, online and distributed modes.
* [LightGBM](https://github.com/Microsoft/LightGBM) - Microsoft's fast, distributed, high performance gradient boosting (GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.
* [libfm](https://github.com/srendle/libfm) - A generic approach that allows to mimic most factorization models by feature engineering.
* [MLDB](https://mldb.ai) - The Machine Learning Database is a database designed for machine learning. Send it commands over a RESTful API to store data, explore it using SQL, then train machine learning models and expose them as APIs.
* [mlpack](https://www.mlpack.org/) - A scalable C++ machine learning library.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [N2D2](https://github.com/CEA-LIST/N2D2) - CEA-List's CAD framework for designing and simulating Deep Neural Network, and building full DNN-based applications on embedded platforms
* [oneDNN](https://github.com/oneapi-src/oneDNN) - An open-source cross-platform performance library for deep learning applications.
* [Opik](https://www.comet.com/site/products/opik/) - Open source engineering platform to debug, evaluate, and monitor your LLM applications, RAG systems, and agentic workflows with comprehensive tracing, automated evaluations, and production-ready dashboards. ([Source Code](https://github.com/comet-ml/opik/))
* [ParaMonte](https://github.com/cdslaborg/paramonte) - A general-purpose library with C/C++ interface for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).
* [proNet-core](https://github.com/cnclabs/proNet-core) - A general-purpose network embedding framework: pair-wise representations optimization Network Edit.
* [PyCaret](https://github.com/pycaret/pycaret) - An open-source, low-code machine learning library in Python that automates machine learning workflows.
* [PyCUDA](https://mathema.tician.de/software/pycuda/) - Python interface to CUDA
* [ROOT](https://root.cern.ch) - A modular scientific software framework. It provides all the functionalities needed to deal with big data processing, statistical analysis, visualization and storage.
* [shark](http://image.diku.dk/shark/sphinx_pages/build/html/index.html) - A fast, modular, feature-rich open-source C++ machine learning library.
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox.
* [sofia-ml](https://code.google.com/archive/p/sofia-ml) - Suite of fast incremental algorithms.
* [Stan](http://mc-stan.org/) - A probabilistic programming language implementing full Bayesian statistical inference with Hamiltonian Monte Carlo sampling.
* [Timbl](https://languagemachines.github.io/timbl/) - A software package/C++ library implementing several memory-based learning algorithms, among which IB1-IG, an implementation of k-nearest neighbor classification, and IGTree, a decision-tree approximation of IB1-IG. Commonly used for NLP.
* [Vowpal Wabbit (VW)](https://github.com/VowpalWabbit/vowpal_wabbit) - A fast out-of-core learning system.
* [Warp-CTC](https://github.com/baidu-research/warp-ctc) - A fast parallel implementation of Connectionist Temporal Classification (CTC), on both CPU and GPU.
* [XGBoost](https://github.com/dmlc/xgboost) - A parallelized optimized general purpose gradient boosting library.
* [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) - A fast library for GBDTs and Random Forests on GPUs.
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) - A fast SVM library on GPUs and CPUs.
* [LKYDeepNN](https://github.com/mosdeo/LKYDeepNN) - A header-only C++11 Neural Network library. Low dependency, native traditional chinese document.
* [xLearn](https://github.com/aksnzhy/xlearn) - A high performance, easy-to-use, and scalable machine learning package, which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data, which is very common in Internet services such as online advertising and recommender systems.
* [Featuretools](https://github.com/featuretools/featuretools) - A library for automated feature engineering. It excels at transforming transactional and relational datasets into feature matrices for machine learning using reusable feature engineering "primitives".
* [skynet](https://github.com/Tyill/skynet) - A library for learning neural networks, has C-interface, net set in JSON. Written in C++ with bindings in Python, C++ and C#.
* [Feast](https://github.com/gojek/feast) - A feature store for the management, discovery, and access of machine learning features. Feast provides a consistent view of feature data for both model training and model serving.
* [Hopsworks](https://github.com/logicalclocks/hopsworks) - A data-intensive platform for AI with the industry's first open-source feature store. The Hopsworks Feature Store provides both a feature warehouse for training and batch based on Apache Hive and a feature serving database, based on MySQL Cluster, for online applications.
* [Polyaxon](https://github.com/polyaxon/polyaxon) - A platform for reproducible and scalable machine learning and deep learning.
* [QuestDB](https://questdb.io/) - A relational column-oriented database designed for real-time analytics on time series and event data.
* [Phoenix](https://phoenix.arize.com) - Uncover insights, surface problems, monitor and fine tune your generative LLM, CV and tabular models.
* [XAD](https://github.com/auto-differentiation/XAD) - Comprehensive backpropagation tool for C++.
* [Truss](https://truss.baseten.co) - An open source framework for packaging and serving ML models.

<a name="cpp-natural-language-processing"></a>
#### Natural Language Processing

* [BLLIP Parser](https://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser).
* [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks. **[Deprecated]**
* [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite is an implementation of Conditional Random Fields (CRFs) for labeling sequential data. **[Deprecated]**
* [frog](https://github.com/LanguageMachines/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
* [libfolia](https://github.com/LanguageMachines/libfolia) - C++ library for the [FoLiA format](https://proycon.github.io/folia/)
* [MeTA](https://github.com/meta-toolkit/meta) - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) is a C++ Data Sciences Toolkit that facilitates mining big text data.
* [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
* [ucto](https://github.com/LanguageMachines/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
* [SentencePiece](https://github.com/google/sentencepiece) - A C++ library for unsupervised text tokenization and detokenization, widely used in modern NLP models.

<a name="cpp-speech-recognition"></a>
#### Speech Recognition
* [Kaldi](https://github.com/kaldi-asr/kaldi) - Kaldi is a toolkit for speech recognition written in C++ and licensed under the Apache License v2.0. Kaldi is intended for use by speech recognition researchers.
* [Vosk](https://github.com/alphacep/vosk-api) - An offline speech recognition toolkit with C++ support, designed for low-resource devices and multiple languages.

<a name="cpp-sequence-analysis"></a>
#### Sequence Analysis
* [ToPS](https://github.com/ayoshiaki/tops) - This is an object-oriented framework that facilitates the integration of probabilistic models for sequences over a user defined alphabet. **[Deprecated]**

<a name="cpp-gesture-detection"></a>
#### Gesture Detection
* [grt](https://github.com/nickgillian/grt) - The Gesture Recognition Toolkit (GRT) is a cross-platform, open-source, C++ machine learning library designed for real-time gesture recognition.

<a name="cpp-reinforcement-learning"></a>
#### Reinforcement Learning
* [RLtools](https://github.com/rl-tools/rl-tools) - The fastest deep reinforcement learning library for continuous control, implemented header-only in pure, dependency-free C++ (Python bindings available as well).

<a name="common-lisp"></a>
## Common Lisp

<a name="common-lisp-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [mgl](https://github.com/melisgl/mgl/) - Neural networks (boltzmann machines, feed-forward and recurrent nets), Gaussian Processes.
* [mgl-gpr](https://github.com/melisgl/mgl-gpr/) - Evolutionary algorithms. **[Deprecated]**
* [cl-libsvm](https://github.com/melisgl/cl-libsvm/) - Wrapper for the libsvm support vector machine library. **[Deprecated]**
* [cl-online-learning](https://github.com/masatoi/cl-online-learning) - Online learning algorithms (Perceptron, AROW, SCW, Logistic Regression).
* [cl-random-forest](https://github.com/masatoi/cl-random-forest) - Implementation of Random Forest in Common Lisp.

<a name="clojure"></a>
## Clojure

<a name="clojure-natural-language-processing"></a>
#### Natural Language Processing

* [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) - Natural Language Processing in Clojure (opennlp).
* [Infections-clj](https://github.com/r0man/inflections-clj) - Rails-like inflection library for Clojure and ClojureScript.

<a name="clojure-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [scicloj.ml](https://github.com/scicloj/scicloj.ml) -  A idiomatic Clojure machine learning library based on tech.ml.dataset with a unique approach for immutable data processing pipelines.
* [clj-ml](https://github.com/joshuaeckroth/clj-ml/) - A machine learning library for Clojure built on top of Weka and friends.
* [clj-boost](https://gitlab.com/alanmarazzi/clj-boost) - Wrapper for XGBoost
* [Touchstone](https://github.com/ptaoussanis/touchstone) - Clojure A/B testing library.
* [Clojush](https://github.com/lspector/Clojush) - The Push programming language and the PushGP genetic programming system implemented in Clojure.
* [lambda-ml](https://github.com/cloudkj/lambda-ml) - Simple, concise implementations of machine learning techniques and utilities in Clojure.
* [Infer](https://github.com/aria42/infer) - Inference and machine learning in Clojure. **[Deprecated]**
* [Encog](https://github.com/jimpil/enclog) - Clojure wrapper for Encog (v3) (Machine-Learning framework that specializes in neural-nets). **[Deprecated]**
* [Fungp](https://github.com/vollmerm/fungp) - A genetic programming library for Clojure. **[Deprecated]**
* [Statistiker](https://github.com/clojurewerkz/statistiker) - Basic Machine Learning algorithms in Clojure. **[Deprecated]**
* [clortex](https://github.com/htm-community/clortex) - General Machine Learning library using Numenta’s Cortical Learning Algorithm. **[Deprecated]**
* [comportex](https://github.com/htm-community/comportex) - Functionally composable Machine Learning library using Numenta’s Cortical Learning Algorithm. **[Deprecated]**

<a name="clojure-deep-learning"></a>
#### Deep Learning
* [MXNet](https://mxnet.apache.org/versions/1.7.0/api/clojure) - Bindings to Apache MXNet - part of the MXNet project
* [Deep Diamond](https://github.com/uncomplicate/deep-diamond) - A fast Clojure Tensor & Deep Learning library
* [jutsu.ai](https://github.com/hswick/jutsu.ai) - Clojure wrapper for deeplearning4j with some added syntactic sugar.
* [cortex](https://github.com/originrose/cortex) - Neural networks, regression and feature learning in Clojure.
* [Flare](https://github.com/aria42/flare) - Dynamic Tensor Graph library in Clojure (think PyTorch, DynNet, etc.)
* [dl4clj](https://github.com/yetanalytics/dl4clj) - Clojure wrapper for Deeplearning4j.

<a name="clojure-data-analysis--data-visualization"></a>
#### Data Analysis
* [tech.ml.dataset](https://github.com/techascent/tech.ml.dataset) - Clojure dataframe library and pipeline for data processing and machine learning
* [Tablecloth](https://github.com/scicloj/tablecloth) - A dataframe grammar wrapping tech.ml.dataset, inspired by several R libraries
* [Panthera](https://github.com/alanmarazzi/panthera) - Clojure API wrapping Python's Pandas library
* [Incanter](http://incanter.org/) - Incanter is a Clojure-based, R-like platform for statistical computing and graphics.
* [PigPen](https://github.com/Netflix/PigPen) - Map-Reduce for Clojure.
* [Geni](https://github.com/zero-one-group/geni) - a Clojure dataframe library that runs on Apache Spark

<a name="clojure-data-visualization"></a>
#### Data Visualization
* [Hanami](https://github.com/jsa-aerial/hanami) : Clojure(Script) library and framework for creating interactive visualization applications based in Vega-Lite (VGL) and/or Vega (VG) specifications. Automatic framing and layouts along with a powerful templating system for abstracting visualization specs
* [Saite](https://github.com/jsa-aerial/saite) -  Clojure(Script) client/server application for dynamic interactive explorations and the creation of live shareable documents capturing them using Vega/Vega-Lite, CodeMirror, markdown, and LaTeX
* [Oz](https://github.com/metasoarous/oz) - Data visualisation using Vega/Vega-Lite and Hiccup, and a live-reload platform for literate-programming
* [Envision](https://github.com/clojurewerkz/envision) - Clojure Data Visualisation library, based on Statistiker and D3.
* [Pink Gorilla Notebook](https://github.com/pink-gorilla/gorilla-notebook) - A Clojure/Clojurescript notebook application/-library based on Gorilla-REPL
* [clojupyter](https://github.com/clojupyter/clojupyter) -  A Jupyter kernel for Clojure - run Clojure code in Jupyter Lab, Notebook and Console.
* [notespace](https://github.com/scicloj/notespace) - Notebook experience in your Clojure namespace
* [Delight](https://github.com/datamechanics/delight) - A listener that streams your spark events logs to delight, a free and improved spark UI

<a name="clojure-interop"></a>
#### Interop

* [Java Interop](https://clojure.org/reference/java_interop) - Clojure has Native Java Interop from which Java's ML ecosystem can be accessed
* [JavaScript Interop](https://clojurescript.org/reference/javascript-api) - ClojureScript has Native JavaScript Interop from which JavaScript's ML ecosystem can be accessed
* [Libpython-clj](https://github.com/clj-python/libpython-clj) - Interop with Python
* [ClojisR](https://github.com/scicloj/clojisr) - Interop with R and Renjin (R on the JVM)

<a name="clojure-misc"></a>
#### Misc
* [Neanderthal](https://neanderthal.uncomplicate.org/) - Fast Clojure Matrix Library (native CPU, GPU, OpenCL, CUDA)
* [kixistats](https://github.com/MastodonC/kixi.stats) - A library of statistical distribution sampling and transducing functions
* [fastmath](https://github.com/generateme/fastmath) - A collection of functions for mathematical and statistical computing, macine learning, etc., wrapping several JVM libraries
* [matlib](https://github.com/atisharma/matlib) - A Clojure library of optimisation and control theory tools and convenience functions based on Neanderthal.

<a name="clojure-extra"></a>
#### Extra
* [Scicloj](https://scicloj.github.io/pages/libraries/) - Curated list of ML related resources for Clojure.

<a name="crystal"></a>
## Crystal

<a name="crystal-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [machine](https://github.com/mathieulaporte/machine) - Simple machine learning algorithm.
* [crystal-fann](https://github.com/NeuraLegion/crystal-fann) - FANN (Fast Artificial Neural Network) binding.

<a name="elixir"></a>
## Elixir

<a name="elixir-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Simple Bayes](https://github.com/fredwu/simple_bayes) - A Simple Bayes / Naive Bayes implementation in Elixir.
* [emel](https://github.com/mrdimosthenis/emel) - A simple and functional machine learning library written in Elixir.
* [Tensorflex](https://github.com/anshuman23/tensorflex) - Tensorflow bindings for the Elixir programming language.

<a name="elixir-natural-language-processing"></a>
#### Natural Language Processing

* [Stemmer](https://github.com/fredwu/stemmer) - An English (Porter2) stemming implementation in Elixir.

<a name="erlang"></a>
## Erlang

<a name="erlang-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Disco](https://github.com/discoproject/disco/) - Map Reduce in Erlang. **[Deprecated]**

<a name="fortran"></a>
## Fortran

<a name="fortran-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [neural-fortran](https://github.com/modern-fortran/neural-fortran) - A parallel neural net microframework.
Read the paper [here](https://arxiv.org/abs/1902.06714).

<a name="fortran-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [ParaMonte](https://github.com/cdslaborg/paramonte) - A general-purpose Fortran library for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).

<a name="go"></a>
## Go

<a name="go-natural-language-processing"></a>
#### Natural Language Processing

* [Cybertron](https://github.com/nlpodyssey/cybertron) - Cybertron: the home planet of the Transformers in Go.
* [snowball](https://github.com/tebeka/snowball) - Snowball Stemmer for Go.
* [word-embedding](https://github.com/ynqa/word-embedding) - Word Embeddings: the full implementation of word2vec, GloVe in Go.
* [sentences](https://github.com/neurosnap/sentences) - Golang implementation of Punkt sentence tokenizer.
* [go-ngram](https://github.com/Lazin/go-ngram) - In-memory n-gram index with compression. *[Deprecated]*
* [paicehusk](https://github.com/Rookii/paicehusk) - Golang implementation of the Paice/Husk Stemming Algorithm. *[Deprecated]*
* [go-porterstemmer](https://github.com/reiver/go-porterstemmer) - A native Go clean room implementation of the Porter Stemming algorithm. **[Deprecated]**

<a name="go-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Spago](https://github.com/nlpodyssey/spago) - Self-contained Machine Learning and Natural Language Processing library in Go.
* [birdland](https://github.com/rlouf/birdland) - A recommendation library in Go.
* [eaopt](https://github.com/MaxHalford/eaopt) - An evolutionary optimization library.
* [leaves](https://github.com/dmitryikh/leaves) - A pure Go implementation of the prediction part of GBRTs, including XGBoost and LightGBM.
* [gobrain](https://github.com/goml/gobrain) - Neural Networks written in Go.
* [go-featureprocessing](https://github.com/nikolaydubina/go-featureprocessing) - Fast and convenient feature processing for low latency machine learning in Go.
* [go-mxnet-predictor](https://github.com/songtianyi/go-mxnet-predictor) - Go binding for MXNet c_predict_api to do inference with a pre-trained model.
* [go-ml-benchmarks](https://github.com/nikolaydubina/go-ml-benchmarks) — benchmarks of machine learning inference for Go.
* [go-ml-transpiler](https://github.com/znly/go-ml-transpiler) - An open source Go transpiler for machine learning models.
* [golearn](https://github.com/sjwhitworth/golearn) - Machine learning for Go.
* [goml](https://github.com/cdipaolo/goml) - Machine learning library written in pure Go.
* [gorgonia](https://github.com/gorgonia/gorgonia) - Deep learning in Go.
* [goro](https://github.com/aunum/goro) - A high-level machine learning library in the vein of Keras.
* [gorse](https://github.com/zhenghaoz/gorse) - An offline recommender system backend based on collaborative filtering written in Go.
* [therfoo](https://github.com/therfoo/therfoo) - An embedded deep learning library for Go.
* [neat](https://github.com/jinyeom/neat) - Plug-and-play, parallel Go framework for NeuroEvolution of Augmenting Topologies (NEAT). **[Deprecated]**
* [go-pr](https://github.com/daviddengcn/go-pr) - Pattern recognition package in Go lang. **[Deprecated]**
* [go-ml](https://github.com/alonsovidales/go_ml) - Linear / Logistic regression, Neural Networks, Collaborative Filtering and Gaussian Multivariate Distribution. **[Deprecated]**
* [GoNN](https://github.com/fxsjy/gonn) - GoNN is an implementation of Neural Network in Go Language, which includes BPNN, RBF, PCN. **[Deprecated]**
* [bayesian](https://github.com/jbrukh/bayesian) - Naive Bayesian Classification for Golang. **[Deprecated]**
* [go-galib](https://github.com/thoj/go-galib) - Genetic Algorithms library written in Go / Golang. **[Deprecated]**
* [Cloudforest](https://github.com/ryanbressler/CloudForest) - Ensembles of decision trees in Go/Golang. **[Deprecated]**
* [go-dnn](https://github.com/sudachen/go-dnn) - Deep Neural Networks for Golang (powered by MXNet)

<a name="go-spatial-analysis-and-geometry"></a>
#### Spatial analysis and geometry

* [go-geom](https://github.com/twpayne/go-geom) - Go library to handle geometries.
* [gogeo](https://github.com/golang/geo) - Spherical geometry in Go.

<a name="go-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [dataframe-go](https://github.com/rocketlaunchr/dataframe-go) - Dataframes for machine-learning and statistics (similar to pandas).
* [gota](https://github.com/go-gota/gota) - Dataframes.
* [gonum/mat](https://godoc.org/gonum.org/v1/gonum/mat) - A linear algebra package for Go.
* [gonum/optimize](https://godoc.org/gonum.org/v1/gonum/optimize) - Implementations of optimization algorithms.
* [gonum/plot](https://godoc.org/gonum.org/v1/plot) - A plotting library.
* [gonum/stat](https://godoc.org/gonum.org/v1/gonum/stat) - A statistics library.
* [SVGo](https://github.com/ajstarks/svgo) - The Go Language library for SVG generation.
* [glot](https://github.com/arafatk/glot) - Glot is a plotting library for Golang built on top of gnuplot.
* [globe](https://github.com/mmcloughlin/globe) - Globe wireframe visualization.
* [gonum/graph](https://godoc.org/gonum.org/v1/gonum/graph) - General-purpose graph library.
* [go-graph](https://github.com/StepLg/go-graph) - Graph library for Go/Golang language. **[Deprecated]**
* [RF](https://github.com/fxsjy/RF.go) - Random forests implementation in Go. **[Deprecated]**

<a name="go-computer-vision"></a>
#### Computer vision

* [GoCV](https://github.com/hybridgroup/gocv) - Package for computer vision using OpenCV 4 and beyond.

<a name="go-reinforcement-learning"></a>
#### Reinforcement learning

* [gold](https://github.com/aunum/gold) - A reinforcement learning library.
* [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) - PyTorch implementations of Stable Baselines (deep) reinforcement learning algorithms.

<a name="haskell"></a>
## Haskell

<a name="haskell-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning
* [haskell-ml](https://github.com/ajtulloch/haskell-ml) - Haskell implementations of various ML algorithms. **[Deprecated]**
* [HLearn](https://github.com/mikeizbicki/HLearn) - a suite of libraries for interpreting machine learning models according to their algebraic structure. **[Deprecated]**
* [hnn](https://github.com/alpmestan/HNN) - Haskell Neural Network library.
* [hopfield-networks](https://github.com/ajtulloch/hopfield-networks) - Hopfield Networks for unsupervised learning in Haskell. **[Deprecated]**
* [DNNGraph](https://github.com/ajtulloch/dnngraph) - A DSL for deep neural networks. **[Deprecated]**
* [LambdaNet](https://github.com/jbarrow/LambdaNet) - Configurable Neural Networks in Haskell. **[Deprecated]**

<a name="java"></a>
## Java

<a name="java-natural-language-processing"></a>
#### Natural Language Processing
* [Cortical.io](https://www.cortical.io/) - Retina: an API performing complex NLP operations (disambiguation, classification, streaming text filtering, etc...) as quickly and intuitively as the brain.
* [IRIS](https://github.com/cortical-io/Iris) - [Cortical.io's](https://cortical.io) FREE NLP, Retina API Analysis Tool (written in JavaFX!) - [See the Tutorial Video](https://www.youtube.com/watch?v=CsF4pd7fGF0).
* [CoreNLP](https://nlp.stanford.edu/software/corenlp.shtml) - Stanford CoreNLP provides a set of natural language analysis tools which can take raw English language text input and give the base forms of words.
* [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml) - A natural language parser is a program that works out the grammatical structure of sentences.
* [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml) - A Part-Of-Speech Tagger (POS Tagger).
* [Stanford Name Entity Recognizer](https://nlp.stanford.edu/software/CRF-NER.shtml) - Stanford NER is a Java implementation of a Named Entity Recognizer.
* [Stanford Word Segmenter](https://nlp.stanford.edu/software/segmenter.shtml) - Tokenization of raw text is a standard pre-processing step for many NLP tasks.
* [Tregex, Tsurgeon and Semgrex](https://nlp.stanford.edu/software/tregex.shtml) - Tregex is a utility for matching patterns in trees, based on tree relationships and regular expression matches on nodes (the name is short for "tree regular expressions").
* [Stanford Phrasal: A Phrase-Based Translation System](https://nlp.stanford.edu/phrasal/)
* [Stanford English Tokenizer](https://nlp.stanford.edu/software/tokenizer.shtml) - Stanford Phrasal is a state-of-the-art statistical phrase-based machine translation system, written in Java.
* [Stanford Tokens Regex](https://nlp.stanford.edu/software/tokensregex.shtml) - A tokenizer divides text into a sequence of tokens, which roughly correspond to "words".
* [Stanford Temporal Tagger](https://nlp.stanford.edu/software/sutime.shtml) - SUTime is a library for recognizing and normalizing time expressions.
* [Stanford SPIED](https://nlp.stanford.edu/software/patternslearning.shtml) - Learning entities from unlabeled text starting with seed sets using patterns in an iterative fashion.
* [Twitter Text Java](https://github.com/twitter/twitter-text/tree/master/java) - A Java implementation of Twitter's text processing library.
* [MALLET](http://mallet.cs.umass.edu/) - A Java-based package for statistical natural language processing, document classification, clustering, topic modelling, information extraction, and other machine learning applications to text.
* [OpenNLP](https://opennlp.apache.org/) - A machine learning based toolkit for the processing of natural language text.
* [LingPipe](http://alias-i.com/lingpipe/index.html) - A tool kit for processing text using computational linguistics.
* [ClearTK](https://github.com/ClearTK/cleartk) - ClearTK provides a framework for developing statistical natural language processing (NLP) components in Java and is built on top of Apache UIMA. **[Deprecated]**
* [Apache cTAKES](https://ctakes.apache.org/) - Apache Clinical Text Analysis and Knowledge Extraction System (cTAKES) is an open-source natural language processing system for information extraction from electronic medical record clinical free-text.
* [NLP4J](https://github.com/emorynlp/nlp4j) - The NLP4J project provides software and resources for natural language processing. The project started at the Center for Computational Language and EducAtion Research, and is currently developed by the Center for Language and Information Research at Emory University. **[Deprecated]**
* [CogcompNLP](https://github.com/CogComp/cogcomp-nlp) - This project collects a number of core libraries for Natural Language Processing (NLP) developed in the University of Illinois' Cognitive Computation Group, for example `illinois-core-utilities` which provides a set of NLP-friendly data structures and a number of NLP-related utilities that support writing NLP applications, running experiments, etc, `illinois-edison` a library for feature extraction from illinois-core-utilities data structures and many other packages.

<a name="java-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [aerosolve](https://github.com/airbnb/aerosolve) - A machine learning library by Airbnb designed from the ground up to be human friendly.
* [AMIDST Toolbox](http://www.amidsttoolbox.com/) - A Java Toolbox for Scalable Probabilistic Machine Learning.
* [Chips-n-Salsa](https://github.com/cicirello/Chips-n-Salsa) - A Java library for genetic algorithms, evolutionary computation, and stochastic local search, with a focus on self-adaptation / self-tuning, as well as parallel execution.
* [Datumbox](https://github.com/datumbox/datumbox-framework) - Machine Learning framework for rapid development of Machine Learning and Statistical applications.
* [ELKI](https://elki-project.github.io/) - Java toolkit for data mining. (unsupervised: clustering, outlier detection etc.)
* [Encog](https://github.com/encog/encog-java-core) - An advanced neural network and machine learning framework. Encog contains classes to create a wide variety of networks, as well as support classes to normalize and process data for these neural networks. Encog trainings using multithreaded resilient propagation. Encog can also make use of a GPU to further speed processing time. A GUI based workbench is also provided to help model and train neural networks.
* [FlinkML in Apache Flink](https://ci.apache.org/projects/flink/flink-docs-master/dev/libs/ml/index.html) - Distributed machine learning library in Flink.
* [H2O](https://github.com/h2oai/h2o-3) - ML engine that supports distributed learning on Hadoop, Spark or your laptop via APIs in R, Python, Scala, REST/JSON.
* [htm.java](https://github.com/numenta/htm.java) - General Machine Learning library using Numenta’s Cortical Learning Algorithm.
* [liblinear-java](https://github.com/bwaldvogel/liblinear-java) - Java version of liblinear.
* [Mahout](https://github.com/apache/mahout) - Distributed machine learning.
* [Meka](http://meka.sourceforge.net/) - An open source implementation of methods for multi-label classification and evaluation (extension to Weka).
* [MLlib in Apache Spark](https://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark.
* [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) - a service for deployment Apache Spark MLLib machine learning models as realtime, batch or reactive web services.
* [Neuroph](http://neuroph.sourceforge.net/) - Neuroph is lightweight Java neural network framework.
* [ORYX](https://github.com/oryxproject/oryx) - Lambda Architecture Framework using Apache Spark and Apache Kafka with a specialization for real-time large-scale machine learning.
* [Samoa](https://samoa.incubator.apache.org/) SAMOA is a framework that includes distributed machine learning for data streams with an interface to plug-in different stream processing platforms.
* [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) - RankLib is a library of learning to rank algorithms. **[Deprecated]**
* [rapaio](https://github.com/padreati/rapaio) - statistics, data mining and machine learning toolbox in Java.
* [RapidMiner](https://rapidminer.com) - RapidMiner integration into Java code.
* [Stanford Classifier](https://nlp.stanford.edu/software/classifier.shtml) - A classifier is a machine learning tool that will take data items and place them into one of k classes.
* [Smile](https://haifengl.github.io/) - Statistical Machine Intelligence & Learning Engine.
* [SystemML](https://github.com/apache/systemml) - flexible, scalable machine learning (ML) language.
* [Tribou](https://tribuo.org) - A machine learning library written in Java by Oracle.
* [Weka](https://www.cs.waikato.ac.nz/ml/weka/) - Weka is a collection of machine learning algorithms for data mining tasks.
* [LBJava](https://github.com/CogComp/lbjava) - Learning Based Java is a modelling language for the rapid development of software systems, offers a convenient, declarative syntax for classifier and constraint definition directly in terms of the objects in the programmer's application.
* [knn-java-library](https://github.com/felipexw/knn-java-library) - Just a simple implementation of K-Nearest Neighbors algorithm using with a bunch of similarity measures.

<a name="java-speech-recognition"></a>
#### Speech Recognition
* [CMU Sphinx](https://cmusphinx.github.io) - Open Source Toolkit For Speech Recognition purely based on Java speech recognition library.

<a name="java-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [Flink](https://flink.apache.org/) - Open source platform for distributed stream and batch data processing.
* [Hadoop](https://github.com/apache/hadoop) - Hadoop/HDFS.
* [Onyx](https://github.com/onyx-platform/onyx) - Distributed, masterless, high performance, fault tolerant data processing. Written entirely in Clojure.
* [Spark](https://github.com/apache/spark) - Spark is a fast and general engine for large-scale data processing.
* [Storm](https://storm.apache.org/) - Storm is a distributed realtime computation system.
* [Impala](https://github.com/cloudera/impala) - Real-time Query for Hadoop.
* [DataMelt](https://jwork.org/dmelt/) - Mathematics software for numeric computation, statistics, symbolic calculations, data analysis and data visualization.
* [Dr. Michael Thomas Flanagan's Java Scientific Library.](https://www.ee.ucl.ac.uk/~mflanaga/java/) **[Deprecated]**

<a name="java-deep-learning"></a>
#### Deep Learning

* [Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j) - Scalable deep learning for industry with parallel GPUs.
* [Keras Beginner Tutorial](https://victorzhou.com/blog/keras-neural-network-tutorial/) - Friendly guide on using Keras to implement a simple Neural Network in Python.
* [deepjavalibrary/djl](https://github.com/deepjavalibrary/djl) - Deep Java Library (DJL) is an open-source, high-level, engine-agnostic Java framework for deep learning, designed to be easy to get started with and simple to use for Java developers.

<a name="javascript"></a>
## JavaScript

<a name="javascript-natural-language-processing"></a>
#### Natural Language Processing

* [Twitter-text](https://github.com/twitter/twitter-text) - A JavaScript implementation of Twitter's text processing library.
* [natural](https://github.com/NaturalNode/natural) - General natural language facilities for node.
* [Knwl.js](https://github.com/loadfive/Knwl.js) - A Natural Language Processor in JS.
* [Retext](https://github.com/retextjs/retext) - Extensible system for analyzing and manipulating natural language.
* [NLP Compromise](https://github.com/spencermountain/compromise) - Natural Language processing in the browser.
* [nlp.js](https://github.com/axa-group/nlp.js) - An NLP library built in node over Natural, with entity extraction, sentiment analysis, automatic language identify, and so more.



<a name="javascript-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [D3.js](https://d3js.org/)
* [High Charts](https://www.highcharts.com/)
* [NVD3.js](http://nvd3.org/)
* [dc.js](https://dc-js.github.io/dc.js/)
* [chartjs](https://www.chartjs.org/)
* [dimple](http://dimplejs.org/)
* [amCharts](https://www.amcharts.com/)
* [D3xter](https://github.com/NathanEpstein/D3xter) - Straight forward plotting built on D3. **[Deprecated]**
* [statkit](https://github.com/rigtorp/statkit) - Statistics kit for JavaScript. **[Deprecated]**
* [datakit](https://github.com/nathanepstein/datakit) - A lightweight framework for data analysis in JavaScript
* [science.js](https://github.com/jasondavies/science.js/) - Scientific and statistical computing in JavaScript. **[Deprecated]**
* [Z3d](https://github.com/NathanEpstein/Z3d) - Easily make interactive 3d plots built on Three.js **[Deprecated]**
* [Sigma.js](http://sigmajs.org/) - JavaScript library dedicated to graph drawing.
* [C3.js](https://c3js.org/) - customizable library based on D3.js for easy chart drawing.
* [Datamaps](https://datamaps.github.io/) - Customizable SVG map/geo visualizations using D3.js. **[Deprecated]**
* [ZingChart](https://www.zingchart.com/) - library written on Vanilla JS for big data visualization.
* [cheminfo](https://www.cheminfo.org/) - Platform for data visualization and analysis, using the [visualizer](https://github.com/npellet/visualizer) project.
* [Learn JS Data](http://learnjsdata.com/)
* [AnyChart](https://www.anychart.com/)
* [FusionCharts](https://www.fusioncharts.com/)
* [Nivo](https://nivo.rocks) - built on top of the awesome d3 and Reactjs libraries


<a name="javascript-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Auto ML](https://github.com/ClimbsRocks/auto_ml) - Automated machine learning, data formatting, ensembling, and hyperparameter optimization for competitions and exploration- just give it a .csv file! **[Deprecated]**
* [Convnet.js](https://cs.stanford.edu/people/karpathy/convnetjs/) - ConvNetJS is a JavaScript library for training Deep Learning models[DEEP LEARNING] **[Deprecated]**
* [Creatify MCP](https://github.com/TSavo/creatify-mcp) - Model Context Protocol server that exposes Creatify AI's video generation capabilities to AI assistants, enabling natural language video creation workflows.
* [Clusterfck](https://harthur.github.io/clusterfck/) - Agglomerative hierarchical clustering implemented in JavaScript for Node.js and the browser. **[Deprecated]**
* [Clustering.js](https://github.com/emilbayes/clustering.js) - Clustering algorithms implemented in JavaScript for Node.js and the browser. **[Deprecated]**
* [Decision Trees](https://github.com/serendipious/nodejs-decision-tree-id3) - NodeJS Implementation of Decision Tree using ID3 Algorithm. **[Deprecated]**
* [DN2A](https://github.com/antoniodeluca/dn2a.js) - Digital Neural Networks Architecture. **[Deprecated]**
* [figue](https://code.google.com/archive/p/figue) - K-means, fuzzy c-means and agglomerative clustering.
* [Gaussian Mixture Model](https://github.com/lukapopijac/gaussian-mixture-model) - Unsupervised machine learning with multivariate Gaussian mixture model.
* [Node-fann](https://github.com/rlidwka/node-fann) - FANN (Fast Artificial Neural Network Library) bindings for Node.js **[Deprecated]**
* [Keras.js](https://github.com/transcranial/keras-js) - Run Keras models in the browser, with GPU support provided by WebGL 2.
* [Kmeans.js](https://github.com/emilbayes/kMeans.js) - Simple JavaScript implementation of the k-means algorithm, for node.js and the browser. **[Deprecated]**
* [LDA.js](https://github.com/primaryobjects/lda) - LDA topic modelling for Node.js
* [Learning.js](https://github.com/yandongliu/learningjs) - JavaScript implementation of logistic regression/c4.5 decision tree **[Deprecated]**
* [machinelearn.js](https://github.com/machinelearnjs/machinelearnjs) - Machine Learning library for the web, Node.js and developers
* [mil-tokyo](https://github.com/mil-tokyo) - List of several machine learning libraries.
* [Node-SVM](https://github.com/nicolaspanel/node-svm) - Support Vector Machine for Node.js
* [Brain](https://github.com/harthur/brain) - Neural networks in JavaScript **[Deprecated]**
* [Brain.js](https://github.com/BrainJS/brain.js) - Neural networks in JavaScript - continued community fork of [Brain](https://github.com/harthur/brain).
* [Bayesian-Bandit](https://github.com/omphalos/bayesian-bandit.js) - Bayesian bandit implementation for Node and the browser. **[Deprecated]**
* [Synaptic](https://github.com/cazala/synaptic) - Architecture-free neural network library for Node.js and the browser.
* [kNear](https://github.com/NathanEpstein/kNear) - JavaScript implementation of the k nearest neighbors algorithm for supervised learning.
* [NeuralN](https://github.com/totemstech/neuraln) - C++ Neural Network library for Node.js. It has advantage on large dataset and multi-threaded training. **[Deprecated]**
* [kalman](https://github.com/itamarwe/kalman) - Kalman filter for JavaScript. **[Deprecated]**
* [shaman](https://github.com/luccastera/shaman) - Node.js library with support for both simple and multiple linear regression. **[Deprecated]**
* [ml.js](https://github.com/mljs/ml) - Machine learning and numerical analysis tools for Node.js and the Browser!
* [ml5](https://github.com/ml5js/ml5-library) - Friendly machine learning for the web!
* [Pavlov.js](https://github.com/NathanEpstein/Pavlov.js) - Reinforcement learning using Markov Decision Processes.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [TensorFlow.js](https://js.tensorflow.org/) - A WebGL accelerated, browser based JavaScript library for training and deploying ML models.
* [JSMLT](https://github.com/jsmlt/jsmlt) - Machine learning toolkit with classification and clustering for Node.js; supports visualization (see [visualml.io](https://visualml.io)).
* [xgboost-node](https://github.com/nuanio/xgboost-node) - Run XGBoost model and make predictions in Node.js.
* [Netron](https://github.com/lutzroeder/netron) - Visualizer for machine learning models.
* [tensor-js](https://github.com/Hoff97/tensorjs) - A deep learning library for the browser, accelerated by WebGL and WebAssembly.
* [WebDNN](https://github.com/mil-tokyo/webdnn) - Fast Deep Neural Network JavaScript Framework. WebDNN uses next generation JavaScript API, WebGPU for GPU execution, and WebAssembly for CPU execution.
* [WebNN](https://webnn.dev) - A new web standard that allows web apps and frameworks to accelerate deep neural networks with on-device hardware such as GPUs, CPUs, or purpose-built AI accelerators.

<a name="javascript-misc"></a>
#### Misc

* [stdlib](https://github.com/stdlib-js/stdlib) - A standard library for JavaScript and Node.js, with an emphasis on numeric computing. The library provides a collection of robust, high performance libraries for mathematics, statistics, streams, utilities, and more.
* [sylvester](https://github.com/jcoglan/sylvester) - Vector and Matrix math for JavaScript. **[Deprecated]**
* [simple-statistics](https://github.com/simple-statistics/simple-statistics) - A JavaScript implementation of descriptive, regression, and inference statistics. Implemented in literate JavaScript with no dependencies, designed to work in all modern browsers (including IE) as well as in Node.js.
* [regression-js](https://github.com/Tom-Alexander/regression-js) - A javascript library containing a collection of least squares fitting methods for finding a trend in a set of data.
* [Lyric](https://github.com/flurry/Lyric) - Linear Regression library. **[Deprecated]**
* [GreatCircle](https://github.com/mwgg/GreatCircle) - Library for calculating great circle distance.
* [MLPleaseHelp](https://github.com/jgreenemi/MLPleaseHelp) - MLPleaseHelp is a simple ML resource search engine. You can use this search engine right now at [https://jgreenemi.github.io/MLPleaseHelp/](https://jgreenemi.github.io/MLPleaseHelp/), provided via GitHub Pages.
* [Pipcook](https://github.com/alibaba/pipcook) - A JavaScript application framework for machine learning and its engineering.

<a name="javascript-demos-and-scripts"></a>
#### Demos and Scripts
* [The Bot](https://github.com/sta-ger/TheBot) - Example of how the neural network learns to predict the angle between two points created with [Synaptic](https://github.com/cazala/synaptic).
* [Half Beer](https://github.com/sta-ger/HalfBeer) - Beer glass classifier created with [Synaptic](https://github.com/cazala/synaptic).
* [NSFWJS](http://nsfwjs.com) - Indecent content checker with TensorFlow.js
* [Rock Paper Scissors](https://rps-tfjs.netlify.com/) - Rock Paper Scissors trained in the browser with TensorFlow.js
* [Heroes Wear Masks](https://heroeswearmasks.fun/) - A fun TensorFlow.js-based oracle that tells, whether one wears a face mask or not. It can even tell when one wears the mask incorrectly.

<a name="julia"></a>
## Julia

<a name="julia-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [MachineLearning](https://github.com/benhamner/MachineLearning.jl) - Julia Machine Learning library. **[Deprecated]**
* [MLBase](https://github.com/JuliaStats/MLBase.jl) - A set of functions to support the development of machine learning algorithms.
* [PGM](https://github.com/JuliaStats/PGM.jl) - A Julia framework for probabilistic graphical models.
* [DA](https://github.com/trthatcher/DiscriminantAnalysis.jl) - Julia package for Regularized Discriminant Analysis.
* [Regression](https://github.com/lindahua/Regression.jl) - Algorithms for regression analysis (e.g. linear regression and logistic regression). **[Deprecated]**
* [Local Regression](https://github.com/JuliaStats/Loess.jl) - Local regression, so smooooth!
* [Naive Bayes](https://github.com/nutsiepully/NaiveBayes.jl) - Simple Naive Bayes implementation in Julia. **[Deprecated]**
* [Mixed Models](https://github.com/dmbates/MixedModels.jl) - A Julia package for fitting (statistical) mixed-effects models.
* [Simple MCMC](https://github.com/fredo-dedup/SimpleMCMC.jl) - basic MCMC sampler implemented in Julia. **[Deprecated]**
* [Distances](https://github.com/JuliaStats/Distances.jl) - Julia module for Distance evaluation.
* [Decision Tree](https://github.com/bensadeghi/DecisionTree.jl) - Decision Tree Classifier and Regressor.
* [Neural](https://github.com/compressed/BackpropNeuralNet.jl) - A neural network in Julia.
* [MCMC](https://github.com/doobwa/MCMC.jl) - MCMC tools for Julia. **[Deprecated]**
* [Mamba](https://github.com/brian-j-smith/Mamba.jl) - Markov chain Monte Carlo (MCMC) for Bayesian analysis in Julia.
* [GLM](https://github.com/JuliaStats/GLM.jl) - Generalized linear models in Julia.
* [Gaussian Processes](https://github.com/STOR-i/GaussianProcesses.jl) - Julia package for Gaussian processes.
* [Online Learning](https://github.com/lendle/OnlineLearning.jl) **[Deprecated]**
* [GLMNet](https://github.com/simonster/GLMNet.jl) - Julia wrapper for fitting Lasso/ElasticNet GLM models using glmnet.
* [Clustering](https://github.com/JuliaStats/Clustering.jl) - Basic functions for clustering data: k-means, dp-means, etc.
* [SVM](https://github.com/JuliaStats/SVM.jl) - SVM for Julia. **[Deprecated]**
* [Kernel Density](https://github.com/JuliaStats/KernelDensity.jl) - Kernel density estimators for Julia.
* [MultivariateStats](https://github.com/JuliaStats/MultivariateStats.jl) - Methods for dimensionality reduction.
* [NMF](https://github.com/JuliaStats/NMF.jl) - A Julia package for non-negative matrix factorization.
* [ANN](https://github.com/EricChiang/ANN.jl) - Julia artificial neural networks. **[Deprecated]**
* [Mocha](https://github.com/pluskid/Mocha.jl) - Deep Learning framework for Julia inspired by Caffe. **[Deprecated]**
* [XGBoost](https://github.com/dmlc/XGBoost.jl) - eXtreme Gradient Boosting Package in Julia.
* [ManifoldLearning](https://github.com/wildart/ManifoldLearning.jl) - A Julia package for manifold learning and nonlinear dimensionality reduction.
* [MXNet](https://github.com/apache/incubator-mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [Merlin](https://github.com/hshindo/Merlin.jl) - Flexible Deep Learning Framework in Julia.
* [ROCAnalysis](https://github.com/davidavdav/ROCAnalysis.jl) - Receiver Operating Characteristics and functions for evaluation probabilistic binary classifiers.
* [GaussianMixtures](https://github.com/davidavdav/GaussianMixtures.jl) - Large scale Gaussian Mixture Models.
* [ScikitLearn](https://github.com/cstjean/ScikitLearn.jl) - Julia implementation of the scikit-learn API.
* [Knet](https://github.com/denizyuret/Knet.jl) - Koç University Deep Learning Framework.
* [Flux](https://fluxml.ai/) - Relax! Flux is the ML library that doesn't make you tensor
* [MLJ](https://github.com/alan-turing-institute/MLJ.jl) - A Julia machine learning framework.
* [CluGen](https://github.com/clugen/CluGen.jl/) - Multidimensional cluster generation in Julia.

<a name="julia-natural-language-processing"></a>
#### Natural Language Processing

* [Topic Models](https://github.com/slycoder/TopicModels.jl) - TopicModels for Julia. **[Deprecated]**
* [Text Analysis](https://github.com/JuliaText/TextAnalysis.jl) - Julia package for text analysis.
* [Word Tokenizers](https://github.com/JuliaText/WordTokenizers.jl) - Tokenizers for Natural Language Processing in Julia
* [Corpus Loaders](https://github.com/JuliaText/CorpusLoaders.jl) - A Julia package providing a variety of loaders for various NLP corpora.
* [Embeddings](https://github.com/JuliaText/Embeddings.jl) - Functions and data dependencies for loading various word embeddings
* [Languages](https://github.com/JuliaText/Languages.jl) - Julia package for working with various human languages
* [WordNet](https://github.com/JuliaText/WordNet.jl) - A Julia package for Princeton's WordNet

<a name="julia-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [Graph Layout](https://github.com/IainNZ/GraphLayout.jl) - Graph layout algorithms in pure Julia.
* [LightGraphs](https://github.com/JuliaGraphs/LightGraphs.jl) - Graph modelling and analysis.
* [Data Frames Meta](https://github.com/JuliaData/DataFramesMeta.jl) - Metaprogramming tools for DataFrames.
* [Julia Data](https://github.com/nfoti/JuliaData) - library for working with tabular data in Julia. **[Deprecated]**
* [Data Read](https://github.com/queryverse/ReadStat.jl) - Read files from Stata, SAS, and SPSS.
* [Hypothesis Tests](https://github.com/JuliaStats/HypothesisTests.jl) - Hypothesis tests for Julia.
* [Gadfly](https://github.com/GiovineItalia/Gadfly.jl) - Crafty statistical graphics for Julia.
* [Stats](https://github.com/JuliaStats/StatsKit.jl) - Statistical tests for Julia.
* [RDataSets](https://github.com/johnmyleswhite/RDatasets.jl) - Julia package for loading many of the data sets available in R.
* [DataFrames](https://github.com/JuliaData/DataFrames.jl) - library for working with tabular data in Julia.
* [Distributions](https://github.com/JuliaStats/Distributions.jl) - A Julia package for probability distributions and associated functions.
* [Data Arrays](https://github.com/JuliaStats/DataArrays.jl) - Data structures that allow missing values. **[Deprecated]**
* [Time Series](https://github.com/JuliaStats/TimeSeries.jl) - Time series toolkit for Julia.
* [Sampling](https://github.com/lindahua/Sampling.jl) - Basic sampling algorithms for Julia.

<a name="julia-misc-stuff--presentations"></a>
#### Misc Stuff / Presentations

* [DSP](https://github.com/JuliaDSP/DSP.jl) - Digital Signal Processing (filtering, periodograms, spectrograms, window functions).
* [JuliaCon Presentations](https://github.com/JuliaCon/presentations) - Presentations for JuliaCon.
* [SignalProcessing](https://github.com/JuliaDSP/DSP.jl) - Signal Processing tools for Julia.
* [Images](https://github.com/JuliaImages/Images.jl) - An image library for Julia.
* [DataDeps](https://github.com/oxinabox/DataDeps.jl) - Reproducible data setup for reproducible science.

<a name="kotlin"></a>
## Kotlin

<a name="kotlin-deep-learning"></a>
#### Deep Learning
* [KotlinDL](https://github.com/JetBrains/KotlinDL) - Deep learning framework written in Kotlin.

<a name="lua"></a>
## Lua

<a name="lua-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Torch7](http://torch.ch/)
  * [cephes](https://github.com/deepmind/torch-cephes) - Cephes mathematical functions library, wrapped for Torch. Provides and wraps the 180+ special mathematical functions from the Cephes mathematical library, developed by Stephen L. Moshier. It is used, among many other places, at the heart of SciPy. **[Deprecated]**
  * [autograd](https://github.com/twitter/torch-autograd) - Autograd automatically differentiates native Torch code. Inspired by the original Python version.
  * [graph](https://github.com/torch/graph) - Graph package for Torch. **[Deprecated]**
  * [randomkit](https://github.com/deepmind/torch-randomkit) - Numpy's randomkit, wrapped for Torch. **[Deprecated]**
  * [signal](https://github.com/soumith/torch-signal) - A signal processing toolbox for Torch-7. FFT, DCT, Hilbert, cepstrums, stft.
  * [nn](https://github.com/torch/nn) - Neural Network package for Torch.
  * [torchnet](https://github.com/torchnet/torchnet) - framework for torch which provides a set of abstractions aiming at encouraging code re-use as well as encouraging modular programming.
  * [nngraph](https://github.com/torch/nngraph) - This package provides graphical computation for nn library in Torch7.
  * [nnx](https://github.com/clementfarabet/lua---nnx) - A completely unstable and experimental package that extends Torch's builtin nn library.
  * [rnn](https://github.com/Element-Research/rnn) - A Recurrent Neural Network library that extends Torch's nn. RNNs, LSTMs, GRUs, BRNNs, BLSTMs, etc.
  * [dpnn](https://github.com/Element-Research/dpnn) - Many useful features that aren't part of the main nn package.
  * [dp](https://github.com/nicholas-leonard/dp) - A deep learning library designed for streamlining research and development using the Torch7 distribution. It emphasizes flexibility through the elegant use of object-oriented design patterns. **[Deprecated]**
  * [optim](https://github.com/torch/optim) - An optimization library for Torch. SGD, Adagrad, Conjugate-Gradient, LBFGS, RProp and more.
  * [unsup](https://github.com/koraykv/unsup) - A package for unsupervised learning in Torch. Provides modules that are compatible with nn (LinearPsd, ConvPsd, AutoEncoder, ...), and self-contained algorithms (k-means, PCA). **[Deprecated]**
  * [manifold](https://github.com/clementfarabet/manifold) - A package to manipulate manifolds.
  * [svm](https://github.com/koraykv/torch-svm) - Torch-SVM library. **[Deprecated]**
  * [lbfgs](https://github.com/clementfarabet/lbfgs) - FFI Wrapper for liblbfgs. **[Deprecated]**
  * [vowpalwabbit](https://github.com/clementfarabet/vowpal_wabbit) - An old vowpalwabbit interface to torch. **[Deprecated]**
  * [OpenGM](https://github.com/clementfarabet/lua---opengm) - OpenGM is a C++ library for graphical modelling, and inference. The Lua bindings provide a simple way of describing graphs, from Lua, and then optimizing them with OpenGM. **[Deprecated]**
  * [spaghetti](https://github.com/MichaelMathieu/lua---spaghetti) - Spaghetti (sparse linear) module for torch7 by @MichaelMathieu **[Deprecated]**
  * [LuaSHKit](https://github.com/ocallaco/LuaSHkit) - A Lua wrapper around the Locality sensitive hashing library SHKit **[Deprecated]**
  * [kernel smoothing](https://github.com/rlowrance/kernel-smoothers) - KNN, kernel-weighted average, local linear regression smoothers. **[Deprecated]**
  * [cutorch](https://github.com/torch/cutorch) - Torch CUDA Implementation.
  * [cunn](https://github.com/torch/cunn) - Torch CUDA Neural Network Implementation.
  * [imgraph](https://github.com/clementfarabet/lua---imgraph) - An image/graph library for Torch. This package provides routines to construct graphs on images, segment them, build trees out of them, and convert them back to images. **[Deprecated]**
  * [videograph](https://github.com/clementfarabet/videograph) - A video/graph library for Torch. This package provides routines to construct graphs on videos, segment them, build trees out of them, and convert them back to videos. **[Deprecated]**
  * [saliency](https://github.com/marcoscoffier/torch-saliency) - code and tools around integral images. A library for finding interest points based on fast integral histograms. **[Deprecated]**
  * [stitch](https://github.com/marcoscoffier/lua---stitch) - allows us to use hugin to stitch images and apply same stitching to a video sequence. **[Deprecated]**
  * [sfm](https://github.com/marcoscoffier/lua---sfm) - A bundle adjustment/structure from motion package. **[Deprecated]**
  * [fex](https://github.com/koraykv/fex) - A package for feature extraction in Torch. Provides SIFT and dSIFT modules. **[Deprecated]**
  * [OverFeat](https://github.com/sermanet/OverFeat) - A state-of-the-art generic dense feature extractor. **[Deprecated]**
  * [wav2letter](https://github.com/facebookresearch/wav2letter) - a simple and efficient end-to-end Automatic Speech Recognition (ASR) system from Facebook AI Research.
* [Numeric Lua](http://numlua.luaforge.net/)
* [Lunatic Python](https://labix.org/lunatic-python)
* [SciLua](http://scilua.org/)
* [Lua - Numerical Algorithms](https://bitbucket.org/lucashnegri/lna) **[Deprecated]**
* [Lunum](https://github.com/jzrake/lunum) **[Deprecated]**
* [Keras GPT Copilot](https://github.com/fabprezja/keras-gpt-copilot) - A python package that integrates an LLM copilot inside the keras model development workflow.

<a name="lua-demos-and-scripts"></a>
#### Demos and Scripts
* [Core torch7 demos repository](https://github.com/e-lab/torch7-demos).
  * linear-regression, logistic-regression
  * face detector (training and detection as separate demos)
  * mst-based-segmenter
  * train-a-digit-classifier
  * train-autoencoder
  * optical flow demo
  * train-on-housenumbers
  * train-on-cifar
  * tracking with deep nets
  * kinect demo
  * filter-bank visualization
  * saliency-networks
* [Training a Convnet for the Galaxy-Zoo Kaggle challenge(CUDA demo)](https://github.com/soumith/galaxyzoo)
* [torch-datasets](https://github.com/rosejn/torch-datasets) - Scripts to load several popular datasets including:
  * BSR 500
  * CIFAR-10
  * COIL
  * Street View House Numbers
  * MNIST
  * NORB
* [Atari2600](https://github.com/fidlej/aledataset) - Scripts to generate a dataset with static frames from the Arcade Learning Environment.



<a name="matlab"></a>
## Matlab

<a name="matlab-computer-vision"></a>
#### Computer Vision

* [Contourlets](http://www.ifp.illinois.edu/~minhdo/software/contourlet_toolbox.tar) - MATLAB source code that implements the contourlet transform and its utility functions.
* [Shearlets](https://www3.math.tu-berlin.de/numerik/www.shearlab.org/software) - MATLAB code for shearlet transform.
* [Curvelets](http://www.curvelet.org/software.html) - The Curvelet transform is a higher dimensional generalization of the Wavelet transform designed to represent images at different scales and different angles.
* [Bandlets](http://www.cmap.polytechnique.fr/~peyre/download/) - MATLAB code for bandlet transform.
* [mexopencv](https://kyamagu.github.io/mexopencv/) - Collection and a development kit of MATLAB mex functions for OpenCV library.

<a name="matlab-natural-language-processing"></a>
#### Natural Language Processing

* [NLP](https://amplab.cs.berkeley.edu/an-nlp-library-for-matlab/) - A NLP library for Matlab.

<a name="matlab-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Training a deep autoencoder or a classifier
on MNIST digits](https://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html) - Training a deep autoencoder or a classifier
on MNIST digits[DEEP LEARNING].
* [Convolutional-Recursive Deep Learning for 3D Object Classification](https://www.socher.org/index.php/Main/Convolutional-RecursiveDeepLearningFor3DObjectClassification) - Convolutional-Recursive Deep Learning for 3D Object Classification[DEEP LEARNING].
* [Spider](https://people.kyb.tuebingen.mpg.de/spider/) - The spider is intended to be a complete object orientated environment for machine learning in Matlab.
* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/#matlab) - A Library for Support Vector Machines.
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) - An Open-Source SVM Library on GPUs and CPUs
* [LibLinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/#download) - A Library for Large Linear Classification.
* [Machine Learning Module](https://github.com/josephmisiti/machine-learning-module) - Class on machine w/ PDF, lectures, code
* [Caffe](https://github.com/BVLC/caffe) - A deep learning framework developed with cleanliness, readability, and speed in mind.
* [Pattern Recognition Toolbox](https://github.com/covartech/PRT) - A complete object-oriented environment for machine learning in Matlab.
* [Pattern Recognition and Machine Learning](https://github.com/PRML/PRMLT) - This package contains the matlab implementation of the algorithms described in the book Pattern Recognition and Machine Learning by C. Bishop.
* [Optunity](https://optunity.readthedocs.io/en/latest/) - A library dedicated to automated hyperparameter optimization with a simple, lightweight API to facilitate drop-in replacement of grid search. Optunity is written in Python but interfaces seamlessly with MATLAB.
* [MXNet](https://github.com/apache/incubator-mxnet/) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [Machine Learning in MatLab/Octave](https://github.com/trekhleb/machine-learning-octave) - Examples of popular machine learning algorithms (neural networks, linear/logistic regressions, K-Means, etc.) with code examples and mathematics behind them being explained.
* [MOCluGen](https://github.com/clugen/MOCluGen/) - Multidimensional cluster generation in MATLAB/Octave.

<a name="matlab-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [ParaMonte](https://github.com/cdslaborg/paramonte) - A general-purpose MATLAB library for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).
* [matlab_bgl](https://www.cs.purdue.edu/homes/dgleich/packages/matlab_bgl/) - MatlabBGL is a Matlab package for working with graphs.
* [gaimc](https://www.mathworks.com/matlabcentral/fileexchange/24134-gaimc---graph-algorithms-in-matlab-code) - Efficient pure-Matlab implementations of graph algorithms to complement MatlabBGL's mex functions.

<a name="net"></a>
## .NET

<a name="net-computer-vision"></a>
#### Computer Vision

* [OpenCVDotNet](https://code.google.com/archive/p/opencvdotnet) - A wrapper for the OpenCV project to be used with .NET applications.
* [Emgu CV](http://www.emgu.com/wiki/index.php/Main_Page) - Cross platform wrapper of OpenCV which can be compiled in Mono to be run on Windows, Linus, Mac OS X, iOS, and Android.
* [AForge.NET](http://www.aforgenet.com/framework/) - Open source C# framework for developers and researchers in the fields of Computer Vision and Artificial Intelligence. Development has now shifted to GitHub.
* [Accord.NET](http://accord-framework.net) - Together with AForge.NET, this library can provide image processing and computer vision algorithms to Windows, Windows RT and Windows Phone. Some components are also available for Java and Android.

<a name="net-natural-language-processing"></a>
#### Natural Language Processing

* [Stanford.NLP for .NET](https://github.com/sergey-tihon/Stanford.NLP.NET/) - A full port of Stanford NLP packages to .NET and also available precompiled as a NuGet package.

<a name="net-general-purpose-machine-learning"></a>
#### General-Purpose Machine Learning

* [Accord-Framework](http://accord-framework.net/) -The Accord.NET Framework is a complete framework for building machine learning, computer vision, computer audition, signal processing and statistical applications.
* [Accord.MachineLearning](https://www.nuget.org/packages/Accord.MachineLearning/) - Support Vector Machines, Decision Trees, Naive Bayesian models, K-means, Gaussian Mixture models and general algorithms such as Ransac, Cross-validation and Grid-Search for machine-learning applications. This package is part of the Accord.NET Framework.
* [DiffSharp](https://diffsharp.github.io/DiffSharp/) - An automatic differentiation (AD) library providing exact and efficient derivatives (gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector products) for machine learning and optimization applications. Operations can be nested to any level, meaning that you can compute exact higher-order derivatives and differentiate functions that are internally making use of differentiation, for applications such as hyperparameter optimization.
* [Encog](https://www.nuget.org/packages/encog-dotnet-core/) - An advanced neural network and machine learning framework. Encog contains classes to create a wide variety of networks, as well as support classes to normalize and process data for these neural networks. Encog trains using multithreaded resilient propagation. Encog can also make use of a GPU to further speed processing time. A GUI based workbench is also provided to help model and train neural networks.
* [GeneticSharp](https://github.com/giacomelli/GeneticSharp) - Multi-platform genetic algorithm library for .NET Core and .NET Framework. The library has several implementations of GA operators, like: selection, crossover, mutation, reinsertion and termination.
* [Infer.NET](https://dotnet.github.io/infer/) - Infer.NET is a framework for running Bayesian inference in graphical models. One can use Infer.NET to solve many different kinds of machine learning problems, from standard problems like classification, recommendation or clustering through customized solutions to domain-specific problems. Infer.NET has been used in a wide variety of domains including information retrieval, bioinformatics, epidemiology, vision, and many others.
* [ML.NET](https://github.com/dotnet/machinelearning) - ML.NET is a cross-platform open-source machine learning framework which makes machine learning accessible to .NET developers. ML.NET was originally developed in Microsoft Research and evolved into a significant framework over the last decade and is used across many product groups in Microsoft like Windows, Bing, PowerPoint, Excel and more.
* [Neural Network Designer](https://sourceforge.net/projects/nnd/) - DBMS management system and designer for neural networks. The designer application is developed using WPF, and is a user interface which allows you to design your neural network, query the network, create and configure chat bots that are capable of asking questions and learning from your feedback. The chat bots can even scrape the internet for information to return in their output as well as to use for learning.
* [Synapses](https://github.com/mrdimosthenis/Synapses) - Neural network library in F#.
* [Vulpes](https://github.com/fsprojects/Vulpes) - Deep belief and deep learning implementation written in F# and leverages CUDA GPU execution with Alea.cuBase.
* [MxNet.Sharp](https://github.com/tech-quantum/MxNet.Sharp) - .NET Standard bindings for Apache MxNet with Imperative, Symbolic and Gluon Interface for developing, training and deploying Machine Learning models in C#. https://mxnet.tech-quantum.com/

<a name="net-data-analysis--data-visualization"></a>
#### Data Analysis / Data Visualization

* [numl](https://www.nuget.org/packages/numl/) - numl is a machine learning library intended to ease the use of using standard modelling techniques for both prediction and clustering.
* [Math.NET Numerics](https://www.nuget.org/packages/MathNet.Numerics/) - Numerical foundation of the Math.NET project, aiming to provide methods and algorithms for numerical computations in science, engineering and everyday use. Supports .Net 4.0, .Net 3.5 and Mono on Windows, Linux and Mac; Silverlight 5, WindowsPhone/SL 8, WindowsPhone 8.1 and Windows 8 with PCL Portable Profiles 47 and 344; Android/iOS with Xamarin.
* [Sho](https://www.microsoft.com/en-us/research/project/sho-the-net-playground-for-data/) - Sho is an interactive environment for data analysis and scientific computing that lets you seamlessly connect scripts (in IronPython) with compiled code (in .NET) to enable fast and flexible prototyping. The environment includes powerful and efficient libraries for linear algebra as well as data visualization that can be used from any .NET language, as well as a feature-rich interactive shell for rapid development.

<a name="objective-c"></a>
## Objective C

<a name="objective-c-general-purpose-machine-learning"></a>
### General-Purpose Machine Learning

* [YCML](https://github.com/yconst/YCML) - A Machine Learning framework for Objective-C and Swift (OS X / iOS).
* [MLPNeuralNet](https://github.com/nikolaypavlov/MLPNeuralNet) - Fast multilayer perceptron neural network library for iOS and Mac OS X. MLPNeuralNet predicts new examples by trained neural networks. It is built on top of the Apple's Accelerate Framework, using vectorized operations and hardware acceleration if available. **[Deprecated]**
* [MAChineLearning](https://github.com/gianlucabertani/MAChineLearning) - An Objective-C multilayer perceptron library, with full support for training through backpropagation. Implemented using vDSP and vecLib, it's 20 times faster than its Java equivalent. Includes sample code for use from Swift.
* [BPN-NeuralNetwork](https://github.com/Kalvar/ios-BPN-NeuralNetwork) - It implemented 3 layers of neural networks ( Input Layer, Hidden Layer and Output Layer ) and it was named Back Propagation Neural Networks (BPN). This network can be used in products recommendation, user behavior analysis, data mining and data analysis. **[Deprecated]**
* [Multi-Perceptron-NeuralNetwork](https://github.com/Kalvar/ios-Multi-Perceptron-NeuralNetwork) - It implemented multi-perceptrons neural network (ニューラルネットワーク) based on Back Propagation Neural Networks (BPN) and designed unlimited-hidden-layers.
* [KRHebbian-Algorithm](https://github.com/Kalvar/ios-KRHebbian-Algorithm) - It is a non-supervisory and self-learning algorithm (adjust the weights) in the neural network of Machine Learning. **[Deprecated]**
* [KRKmeans-Algorithm](https://github.com/Kalvar/ios-KRKmeans-Algorithm) - It implemented K-Means  clustering and classification algorithm. It could be used in data mining and image compression. **[Deprecated]**
* [KRFuzzyCMeans-Algorithm](https://github.com/Kalvar/ios-KRFuzzyCMeans-Algorithm) - It implemented Fuzzy C-Means (FCM) the fuzzy clustering / classification algorithm on Machine Learning. It could be used in data mining and image compression. **[Deprecated]**

<a name="ocaml"></a>
## OCaml

<a name="ocaml-general-purpose-machine-learning"></a>
### General-Purpose Machine Learning

* [Oml](https://github.com/rleonid/oml) - A general statistics and machine learning library.
* [GPR](https://mmottl.github.io/gpr/) - Efficient Gaussian Process Regression in OCaml.
* [Libra-Tk](https://libra.cs.uoregon.edu) - Algorithms for learning and inference with discrete probabilistic models.
* [TensorFlow](https://github.com/LaurentMazare/tensorflow-ocaml) - OCaml bindings for TensorFlow.

<a name="opencv"></a>
## OpenCV

<a name="opencv-ComputerVision and Text Detection"></a>
### OpenSource-Computer-Vision

* [OpenCV](https://github.com/opencv/opencv) - A OpenSource Computer Vision Library

<a name="perl"></a>
## Perl

<a name="perl-data-analysis--data-visualization"></a>
### Data Analysis / Data Visualization

* [Perl Data Language](https://metacpan.org/pod/Paws::MachineLearning), a pluggable architecture for data and image processing, which can
be [used for machine learning](https://github.com/zenogantner/PDL-ML).

<a name="perl-general-purpose-machine-learning"></a>
### General-Purpose Machine Learning

* [MXnet for Deep Learning, in Perl](https://github.com/apache/incubator-mxnet/tree/master/perl-package),
also [released in CPAN](https://metacpan.org/pod/AI::MXNet).
* [Perl Data Language](https://metacpan.org/pod/Paws::MachineLearning),
using AWS machine learning platform from Perl.
* [Algorithm::SVMLight](https://metacpan.org/pod/Algorithm::SVMLight),
  implementation of Support Vector Machines with SVMLight under it. **[Deprecated]**
* Several machine learning and artificial intelligence models are
  included in the [`AI`](https://metacpan.org/search?size=20&q=AI)
  namespace. For instance, you can
  find [Naïve Bayes](https://metacpan.org/pod/AI::NaiveBayes).

<a name="perl6"></a>
## Perl 6

* [Support Vector Machines](https://github.com/titsuki/p6-Algorithm-LibSVM)
* [Naïve Bayes](https://github.com/titsuki/p6-Algorithm-NaiveBayes)

<a name="perl-6-data-analysis--data-visualization"></a>
### Data Analysis / Data Visualization

* [Perl Data Language](https://metacpan.org/pod/Paws::MachineLearning),
a pluggable architecture for data and image processing, which can
be
[used for machine learning](https://github.com/zenogantner/PDL-ML).

<a name="perl-6-general-purpose-machine-learning"></a>
### General-Purpose Machine Learning

<a name="php"></a>
## PHP

<a name="php-natural-language-processing"></a>
### Natural Language Processing

* [jieba-php](https://github.com/fukuball/jieba-php) - Chinese Words Segmentation Utilities.

<a name="php-general-purpose-machine-learning"></a>
### General-Purpose Machine Learning

* [PHP-ML](https://gitlab.com/php-ai/php-ml) - Machine Learning library for PHP. Algorithms, Cross Validation, Neural Network, Preprocessing, Feature Extraction and much more in one library.
* [PredictionBuilder](https://github.com/denissimon/prediction-builder) - A library for machine learning that builds predictions using a linear regression.
* [Rubix ML](https://github.com/RubixML) - A high-level machine learning (ML) library that lets you build programs that learn from data using the PHP language.
* [19 Questions](https://github.com/fulldecent/19-questions) - A machine learning / bayesian inference assigning attributes to objects.

<a name="python"></a>
## Python

<a name="python-computer-vision"></a>
#### Computer Vision

* [LightlyTrain](https://github.com/lightly-ai/lightly-train) - Pretrain computer vision models on unlabeled data for industrial applications
* [Scikit-Image](https://github.com/scikit-image/scikit-image) - A collection of algorithms for image processing in Python.
* [Scikit-Opt](https://github.com/guofei9987/scikit-opt) - Swarm Intelligence in Python (Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing, Ant Colony Algorithm, Immune Algorithm, Artificial Fish Swarm Algorithm in Python)
* [SimpleCV](http://simplecv.org/) - An open source computer vision framework that gives access to several high-powered computer vision libraries, such as OpenCV. Written on Python and runs on Mac, Windows, and Ubuntu Linux.
* [Vigranumpy](https://github.com/ukoethe/vigra) - Python bindings for the VIGRA C++ computer vision library.
* [OpenFace](https://cmusatyalab.github.io/openface/) - Free and open source face recognition with deep neural networks.
* [PCV](https://github.com/jesolem/PCV) - Open source Python module for computer vision. **[Deprecated]**
* [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library that recognizes and manipulates faces from Python or from the command line.
* [deepface](https://github.com/serengil/deepface) - A lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for Python covering cutting-edge models such as VGG-Face, FaceNet, OpenFace, DeepFace, DeepID, Dlib and ArcFace.
* [retinaface](https://github.com/serengil/retinaface) - deep learning based cutting-edge facial detector for Python coming with facial landmarks
* [dockerface](https://github.com/natanielruiz/dockerface) - Easy to install and use deep learning Faster R-CNN face detection for images and video in a docker container. **[Deprecated]**
* [Detectron](https://github.com/facebookresearch/Detectron) - FAIR's software system that implements state-of-the-art object detection algorithms, including Mask R-CNN. It is written in Python and powered by the Caffe2 deep learning framework. **[Deprecated]**
* [detectron2](https://github.com/facebookresearch/detectron2) - FAIR's next-generation research platform for object detection and segmentation. It is a ground-up rewrite of the previous version, Detectron, and is powered by the PyTorch deep learning framework.
* [albumentations](https://github.com/albu/albumentations) - А fast and framework agnostic image augmentation library that implements a diverse set of augmentation techniques. Supports classification, segmentation, detection out of the box. Was used to win a number of Deep Learning competitions at Kaggle, Topcoder and those that were a part of the CVPR workshops.
* [pytessarct](https://github.com/madmaze/pytesseract) - Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and "read" the text embedded in images. Python-tesseract is a wrapper for [Google's Tesseract-OCR Engine](https://github.com/tesseract-ocr/tesseract).
* [imutils](https://github.com/jrosebr1/imutils) - A library containing Convenience functions to make basic image processing operations such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV and Python.
* [PyTorchCV](https://github.com/donnyyou/PyTorchCV) - A PyTorch-Based Framework for Deep Learning in Computer Vision.
* [joliGEN](https://github.com/jolibrain/joliGEN) - Generative AI Image Toolset with GANs and Diffusion for Real-World Applications.
* [Self-supervised learning](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html)
* [neural-style-pt](https://github.com/ProGamerGov/neural-style-pt) - A PyTorch implementation of Justin Johnson's neural-style (neural style transfer).
* [Detecto](https://github.com/alankbi/detecto) - Train and run a computer vision model with 5-10 lines of code.
* [neural-dream](https://github.com/ProGamerGov/neural-dream) - A PyTorch implementation of DeepDream.
* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) - A real-time multi-person keypoint detection library for body, face, hands, and foot estimation
* [Deep High-Resolution-Net](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) - A PyTorch implementation of CVPR2019 paper "Deep High-Resolution Representation Learning for Human Pose Estimation"
* [TF-GAN](https://github.com/tensorflow/gan) - TF-GAN is a lightweight library for training and evaluating Generative Adversarial Networks (GANs).
* [dream-creator](https://github.com/ProGamerGov/dream-creator) - A PyTorch implementation of DeepDream. Allows individuals to quickly and easily train their own custom GoogleNet models with custom datasets for DeepDream.
* [Lucent](https://github.com/greentfrapp/lucent) - Tensorflow and OpenAI Clarity's Lucid adapted for PyTorch.
* [lightly](https://github.com/lightly-ai/lightly) - Lightly is a computer vision framework for self-supervised learning.
* [Learnergy](https://github.com/gugarosa/learnergy) - Energy-based machine learning models built upon PyTorch.
* [OpenVisionAPI](https://github.com/openvisionapi) - Open source computer vision API based on open source models.
* [IoT Owl](https://github.com/Ret2Me/IoT-Owl) - Light face detection and recognition system with huge possibilities, based on Microsoft Face API and TensorFlow made for small IoT devices like raspberry pi.
* [Exadel CompreFace](https://github.com/exadel-inc/CompreFace) - face recognition system that can be easily integrated into any system without prior machine learning skills. CompreFace provides REST API for face recognition, face verification, face detection, face mask detection, landmark detection, age, and gender recognition and is easily deployed with docker.
* [computer-vision-in-action](https://github.com/Charmve/computer-vision-in-action) - as known as ``L0CV``, is a new generation of computer vision open source online learning media, a cross-platform interactive learning framework integrating graphics, source code and HTML. the L0CV ecosystem — Notebook, Datasets, Source Code, and from Diving-in to Advanced — as well as the L0CV Hub.
* [timm](https://github.com/rwightman/pytorch-image-models) - PyTorch image models, scripts, pretrained weights -- ResNet, ResNeXT, EfficientNet, EfficientNetV2, NFNet, Vision Transformer, MixNet, MobileNet-V3/V2, RegNet, DPN, CSPNet, and more.
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) - A PyTorch-based toolkit that offers pre-trained segmentation models for computer vision tasks. It simplifies the development of image segmentation applications by providing a collection of popular architecture implementations, such as UNet and PSPNet, along with pre-trained weights, making it easier for researchers and developers to achieve high-quality pixel-level object segmentation in images.
* [segmentation_models](https://github.com/qubvel/segmentation_models) - A TensorFlow Keras-based toolkit that offers pre-trained segmentation models for computer vision tasks. It simplifies the development of image segmentation applications by providing a collection of popular architecture implementations, such as UNet and PSPNet, along with pre-trained weights, making it easier for researchers and developers to achieve high-quality pixel-level object segmentation in images.
* [MLX](https://github.com/ml-explore/mlx)- MLX is an array framework for machine learning on Apple silicon, developed by Apple machine learning research.

<a name="python-natural-language-processing"></a>
#### Natural Language Processing

* [pkuseg-python](https://github.com/lancopku/pkuseg-python) - A better version of Jieba, developed by Peking University.
* [NLTK](https://www.nltk.org/) - A leading platform for building Python programs to work with human language data.
* [Pattern](https://github.com/clips/pattern) - A web mining module for the Python programming language. It has tools for natural language processing, machine learning, among others.
* [Quepy](https://github.com/machinalis/quepy) - A python framework to transform natural language questions to queries in a database query language.
* [TextBlob](http://textblob.readthedocs.io/en/dev/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of NLTK and Pattern, and plays nicely with both.
* [YAlign](https://github.com/machinalis/yalign) - A sentence aligner, a friendly tool for extracting parallel sentences from comparable corpora. **[Deprecated]**
* [jieba](https://github.com/fxsjy/jieba#jieba-1) - Chinese Words Segmentation Utilities.
* [SnowNLP](https://github.com/isnowfy/snownlp) - A library for processing Chinese text.
* [spammy](https://github.com/tasdikrahman/spammy) - A library for email Spam filtering built on top of NLTK
* [loso](https://github.com/fangpenlin/loso) - Another Chinese segmentation library. **[Deprecated]**
* [genius](https://github.com/duanhongyi/genius) - A Chinese segment based on Conditional Random Field.
* [KoNLPy](http://konlpy.org) - A Python package for Korean natural language processing.
* [nut](https://github.com/pprett/nut) - Natural language Understanding Toolkit. **[Deprecated]**
* [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)
* [BLLIP Parser](https://pypi.org/project/bllipparser/) - Python bindings for the BLLIP Natural Language Parser (also known as the Charniak-Johnson parser). **[Deprecated]**
* [PyNLPl](https://github.com/proycon/pynlpl) - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](https://proycon.github.io/folia/), but also ARPA language models,

---

# TRANSFERLEARNING

**Source:** https://github.com/jindongwang/transferlearning
**Stars:** 14,000+
**Resources:** 151+

## Content

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<h1 align="center">
  <br>
  <img src="png/logo.jpg" alt="Transfer Leanring" width="500">
</h1>

<h4 align="center">Everything about Transfer Learning. 迁移学习.</h4>

<p align="center">
  <strong><a href="#0papers-论文">Papers</a></strong> •
  <strong><a href="#1introduction-and-tutorials-简介与教程">Tutorials</a></strong> •
  <a href="#2transfer-learning-areas-and-papers-研究领域与相关论文">Research areas</a> •
  <a href="#3theory-and-survey-理论与综述">Theory</a> •
  <a href="#3theory-and-survey-理论与综述">Survey</a> •
  <strong><a href="https://github.com/jindongwang/transferlearning/tree/master/code">Code</a></strong> •
  <strong><a href="#7datasets-and-benchmarks-数据集与评测结果">Dataset & benchmark</a></strong>
</p>
<p align="center">
  <a href="#6transfer-learning-thesis-硕博士论文">Thesis</a> •
  <a href="#5transfer-learning-scholars-著名学者">Scholars</a> •
  <a href="#8transfer-learning-challenges-迁移学习比赛">Contests</a> •
  <a href="#journals-and-conferences">Journal/conference</a> •
  <a href="#applications-迁移学习应用">Applications</a> •
  <a href="#other-resources-其他资源">Others</a> •
  <a href="#contributing-欢迎参与贡献">Contributing</a>
</p>

**Widely used by top conferences and journals:** 
- Conferences: [[CVPR'22](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/html/Zhang_Segmenting_Across_Places_The_Need_for_Fair_Transfer_Learning_With_CVPRW_2022_paper.html)] [[NeurIPS'21](https://proceedings.neurips.cc/paper/2021/file/731b03008e834f92a03085ef47061c4a-Paper.pdf)] [[IJCAI'21](https://arxiv.org/abs/2103.03097)] [[ESEC/FSE'20](https://dl.acm.org/doi/abs/10.1145/3368089.3409696)] [[IJCNN'20](https://ieeexplore.ieee.org/abstract/document/9207556)] [[ACMMM'18](https://dl.acm.org/doi/abs/10.1145/3240508.3240512)] [[ICME'19](https://ieeexplore.ieee.org/abstract/document/8784776/)]
- Journals: [[IEEE TKDE](https://ieeexplore.ieee.org/abstract/document/9782500/)] [[ACM TIST](https://dl.acm.org/doi/abs/10.1145/3360309)] [[Information sciences](https://www.sciencedirect.com/science/article/pii/S0020025520308458)] [[Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231221007025)] [[IEEE Transactions on Cognitive and Developmental Systems](https://ieeexplore.ieee.org/abstract/document/9659817)]

```
@Misc{transferlearning.xyz,
howpublished = {\url{http://transferlearning.xyz}},   
title = {Everything about Transfer Learning and Domain Adapation},  
author = {Wang, Jindong and others}  
}  
```

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE) [![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu) 

Related Codes:
  - Large language model evaluation: [[llm-eval](https://llm-eval.github.io/)]
  - Large language model enhancement: [[llm-enhance](https://llm-enhance.github.io/)]
  - Robust machine learning: [[robustlearn: robust machine learning](https://github.com/microsoft/robustlearn)]
  - Semi-supervised learning: [[USB: unified semi-supervised learning benchmark](https://github.com/microsoft/Semi-supervised-learning)] | [[TorchSSL: a unified SSL library](https://github.com/TorchSSL/TorchSSL)] 
  - LLM benchmark: [[PromptBench: adversarial robustness of prompts of LLMs](https://github.com/microsoft/promptbench)]
  - Federated learning: [[PersonalizedFL: library for personalized federated learning](https://github.com/microsoft/PersonalizedFL)]
  - Activity recognition and machine learning [[Activity recognition](https://github.com/jindongwang/activityrecognition)]｜[[Machine learning](https://github.com/jindongwang/MachineLearning)]

- - -

**NOTE:** You can directly open the code in [Gihub Codespaces](https://docs.github.com/en/codespaces/getting-started/quickstart#introduction) on the web to run them without downloading! Also, try [github.dev](https://github.dev/jindongwang/transferlearning).

## 0.Papers (论文)

[Awesome transfer learning papers (迁移学习文章汇总)](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- [Paperweekly](http://www.paperweekly.site/collections/231/papers): A website to recommend and read paper notes

**Latest papers**: 

- By topic: [doc/awesome_papers.md](/doc/awesome_paper.md)
- By date: [doc/awesome_paper_date.md](/doc/awesome_paper_date.md)

*Updated at 2024-02-18:*

- Simulations of Common Unsupervised Domain Adaptation Algorithms for Image Classification [[arxiv](https://arxiv.org/abs/2502.10694)]
  - Unsupervised domain adaptaiton for image classification

- Semantics-aware Test-time Adaptation for 3D Human Pose Estimation [[arxiv](https://arxiv.org/abs/2502.10724)]
  - Test-time adaptation for3D human pose estimation

- Transfer Learning of CATE with Kernel Ridge Regression [[arxiv](https://arxiv.org/abs/2502.11331)]
  - Transfer learning with kernel ridge regression

- Why Domain Generalization Fail? A View of Necessity and Sufficiency [[arxiv](https://arxiv.org/abs/2502.10716)] 
  - Analyze why domain generalization fail from the view of necessity and sufficiency


*Updated at 2024-02-11:*

- Beyond Batch Learning: Global Awareness Enhanced Domain Adaptation [[arxiv](https://arxiv.org/abs/2502.06272)]
  - Global awareness for enhanced domain adaptation

- - -

## 1.Introduction and Tutorials (简介与教程)

Want to quickly learn transfer learning？想尽快入门迁移学习？看下面的教程。

- Books 书籍
  - **Introduction to Transfer Learning: Algorithms and Practice** [[Buy or read](https://link.springer.com/book/9789811975837)]
  - **《迁移学习》（杨强）** [[Buy](https://item.jd.com/12930984.html)] [[English version](https://www.cambridge.org/core/books/transfer-learning/CCFFAFE3CDBC245047F1DEC71D9EF3C7)]
  - **《迁移学习导论》(王晋东、陈益强著)** [[Homepage](http://jd92.wang/tlbook)] [[Buy](https://item.jd.com/13272157.html)]

- Blogs 博客
  - [Zhihu blogs - 知乎专栏《小王爱迁移》系列文章](https://zhuanlan.zhihu.com/p/130244395)
	
- Video tutorials 视频教程
  - Transfer learning 迁移学习:
    - [Recent advance of transfer learning - 2022年最新迁移学习发展现状探讨](https://www.bilibili.com/video/BV1nY411E7Uc/)
    - [Definitions of transfer learning area - 迁移学习领域名词解释](https://www.bilibili.com/video/BV1fu411o7BW) [[Article](https://zhuanlan.zhihu.com/p/428097044)]
    - [Transfer learning by Hung-yi Lee @ NTU - 台湾大学李宏毅的视频讲解(中文视频)](https://www.youtube.com/watch?v=qD6iD4TFsdQ)
  - Domain generalization 领域泛化：
    - [IJCAI-ECAI'22 tutorial on domain generalization - 领域泛化tutorial](https://dgresearch.github.io/)
    - [Domain generalization - 迁移学习新兴研究方向领域泛化](https://www.bilibili.com/video/BV1ro4y1S7dd/)
  - Domain adaptation 领域自适应：
    - [Domain adaptation - 迁移学习中的领域自适应方法(中文)](https://www.bilibili.com/video/BV1T7411R75a/) 
  

- Brief introduction and slides 简介与ppt资料
  - [Recent advance of transfer learning](https://jd92.wang/assets/files/l16_aitime.pdf)
  - [Domain generalization survey](http://jd92.wang/assets/files/DGSurvey-ppt.pdf)
  - [Brief introduction in Chinese](https://github.com/jindongwang/transferlearning/blob/master/doc/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%AE%80%E4%BB%8B.md)
	- [PPT (English)](http://jd92.wang/assets/files/l03_transferlearning.pdf) | [PPT (中文)](http://jd92.wang/assets/files/l08_tl_zh.pdf)
  - 迁移学习中的领域自适应方法 Domain adaptation: [PDF](http://jd92.wang/assets/files/l12_da.pdf) ｜ [Video on Bilibili](https://www.bilibili.com/video/BV1T7411R75a/) | [Video on Youtube](https://www.youtube.com/watch?v=RbIsHNtluwQ&t=22s)
  - Tutorial on transfer learning by Qiang Yang: [IJCAI'13](http://ijcai13.org/files/tutorial_slides/td2.pdf) | [2016 version](http://kddchina.org/file/IntroTL2016.pdf)

- Talk is cheap, show me the code 动手教程、代码、数据 
  - [Pytorch tutorial on transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
	- [Pytorch finetune](https://github.com/jindongwang/transferlearning/tree/master/code/AlexNet_ResNet)
	- [DeepDA: a unified deep domain adaptation toolbox](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA)
	- [DeepDG: a unified deep domain generalization toolbox](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG)
	- [更多 More...](https://github.com/jindongwang/transferlearning/tree/master/code)

- [Transfer Learning Scholars and Labs - 迁移学习领域的著名学者、代表工作及实验室介绍](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)
- [Negative transfer - 负迁移](https://www.zhihu.com/question/66492194/answer/242870418)

- - -

## 2.Transfer Learning Areas and Papers (研究领域与相关论文)

- [Survey](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#survey)
- [Theory](#theory)
- [Per-training/Finetuning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#per-trainingfinetuning)
- [Knowledge distillation](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#knowledge-distillation)
- [Traditional domain adaptation](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#traditional-domain-adaptation)
- [Deep domain adaptation](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#deep-domain-adaptation)
- [Domain generalization](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#domain-generalization)
- [Source-free domain adaptation](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#source-free-domain-adaptation)
- [Multi-source domain adaptation](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#multi-source-domain-adaptation)
- [Heterogeneous transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#heterogeneous-transfer-learning)
- [Online transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#online-transfer-learning)
- [Zero-shot / few-shot learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#zero-shot--few-shot-learning)
- [Multi-task learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#multi-task-learning)
- [Transfer reinforcement learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-reinforcement-learning)
- [Transfer metric learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-metric-learning)
- [Federated transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#federated-transfer-learning)
- [Lifelong transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#lifelong-transfer-learning)
- [Safe transfer learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#safe-transfer-learning)
- [Transfer learning applications](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-learning-applications)

- - -

## 3.Theory and Survey (理论与综述)

Here are some articles on transfer learning theory and survey.

**Survey (综述文章)：**

- 2023 Source-Free Unsupervised Domain Adaptation: A Survey [[arxiv](http://arxiv.org/abs/2301.00265)]
- 2022 [Transfer Learning for Future Wireless Networks: A Comprehensive Survey](https://arxiv.org/abs/2102.07572)
- 2022 [A Review of Deep Transfer Learning and Recent Advancements](https://arxiv.org/abs/2201.09679)
- 2022 [Transferability in Deep Learning: A Survey](https://paperswithcode.com/paper/transferability-in-deep-learning-a-survey), from Mingsheng Long in THU.
- 2021 Domain generalization: IJCAI-21 [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://arxiv.org/abs/2103.03097) | [知乎文章](https://zhuanlan.zhihu.com/p/354740610) | [微信公众号](https://mp.weixin.qq.com/s/DsoVDYqLB1N7gj9X5UnYqw)
  - First survey on domain generalization
  - 第一篇对Domain generalization (领域泛化)的综述
- 2021 Vision-based activity recognition: [A Survey of Vision-Based Transfer Learning in Human Activity Recognition](https://www.mdpi.com/2079-9292/10/19/2412)
- 2021 ICSAI [A State-of-the-Art Survey of Transfer Learning in Structural Health Monitoring](https://ieeexplore.ieee.org/abstract/document/9664171)
- 2020 [Transfer learning: survey and classification](https://link.springer.com/chapter/10.1007/978-981-15-5345-5_13), Advances in Intelligent Systems and Computing. 
- 2020 迁移学习最新survey，来自中科院计算所庄福振团队，发表在Proceedings of the IEEE: [A Comprehensive Survey on Transfer Learning](https://arxiv.org/abs/1911.02685)
- 2020 负迁移的综述：[Overcoming Negative Transfer: A Survey](https://arxiv.org/abs/2009.00909)
- 2020 知识蒸馏的综述: [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
- 用transfer learning进行sentiment classification的综述：[A Survey of Sentiment Analysis Based on Transfer Learning](https://ieeexplore.ieee.org/abstract/document/8746210) 
- 2019 一篇新survey：[Transfer Adaptation Learning: A Decade Survey](https://arxiv.org/abs/1903.04687)
- 2018 一篇迁移度量学习的综述: [Transfer Metric Learning: Algorithms, Applications and Outlooks](https://arxiv.org/abs/1810.03944)
- 2018 一篇最近的非对称情况下的异构迁移学习综述：[Asymmetric Heterogeneous Transfer Learning: A Survey](https://arxiv.org/abs/1804.10834)
- 2018 Neural style transfer的一个survey：[Neural Style Transfer: A Review](https://arxiv.org/abs/1705.04058)
- 2018 深度domain adaptation的一个综述：[Deep Visual Domain Adaptation: A Survey](https://www.sciencedirect.com/science/article/pii/S0925231218306684)
- 2017 多任务学习的综述，来自香港科技大学杨强团队：[A survey on multi-task learning](https://arxiv.org/abs/1707.08114)
- 2017 异构迁移学习的综述：[A survey on heterogeneous transfer learning](https://link.springer.com/article/10.1186/s40537-017-0089-0)
- 2017 跨领域数据识别的综述：[Cross-dataset recognition: a survey](https://arxiv.org/abs/1705.04396)
- 2016 [A survey of transfer learning](https://pan.baidu.com/s/1gfgXLXT)。其中交代了一些比较经典的如同构、异构等学习方法代表性文章。
- 2015 中文综述：[迁移学习研究进展](https://pan.baidu.com/s/1bpautob)
- 2010 [A survey on transfer learning](http://ieeexplore.ieee.org/abstract/document/5288526/)
- Survey on applications - 应用导向的综述：
	- 视觉domain adaptation综述：[Visual Domain Adaptation: A Survey of Recent Advances](https://pan.baidu.com/s/1o8BR7Vc)
	- 迁移学习应用于行为识别综述：[Transfer Learning for Activity Recognition: A Survey](https://pan.baidu.com/s/1kVABOYr)
	- 迁移学习与增强学习：[Transfer Learning for Reinforcement Learning Domains: A Survey](https://pan.baidu.com/s/1slfr0w1)
	- 多个源域进行迁移的综述：[A Survey of Multi-source Domain Adaptation](https://pan.baidu.com/s/1eSGREF4)。

**Theory （理论文章）:**

- ICML-20 [Few-shot domain adaptation by causal mechanism transfer](https://arxiv.org/pdf/2002.03497.pdf)
	- The first work on causal transfer learning
	- 日本理论组大佬Sugiyama的工作，causal transfer learning
- CVPR-19 [Characterizing and Avoiding Negative Transfer](https://arxiv.org/abs/1811.09751)
	- Characterizing and avoid negative transfer
	- 形式化并提出如何避免负迁移
- ICML-20 [On Learning Language-Invariant Representations for Universal Machine Translation](https://arxiv.org/abs/2008.04510)
  - Theory for universal machine translation
  - 对统一机器翻译模型进行了理论论证
- NIPS-06 [Analysis of Representations for Domain Adaptation](https://dl.acm.org/citation.cfm?id=2976474)
- ML-10 [A Theory of Learning from Different Domains](https://link.springer.com/article/10.1007/s10994-009-5152-4)
- NIPS-08 [Learning Bounds for Domain Adaptation](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)
- COLT-09 [Domain adaptation: Learning bounds and algorithms](https://arxiv.org/abs/0902.3430)
- MMD paper：[A Hilbert Space Embedding for Distributions](https://link.springer.com/chapter/10.1007/978-3-540-75225-7_5) and [A Kernel Two-Sample Test](http://www.jmlr.org/papers/v13/gretton12a.html)
- Multi-kernel MMD paper: [Optimal kernel choice for large-scale two-sample tests](http://papers.nips.cc/paper/4727-optimal-kernel-choice-for-large-scale-two-sample-tests)

_ _ _

## 4.Code (代码)

Unified codebases for:
- [Deep domain adaptation](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA)
- [Deep domain generalization](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG)
- See all codes here: https://github.com/jindongwang/transferlearning/tree/master/code.

More: see [HERE](https://github.com/jindongwang/transferlearning/tree/master/code) and [HERE](https://colab.research.google.com/drive/1MVuk95mMg4ecGyUAIG94vedF81HtWQAr?usp=sharing) for an instant run using Google's Colab.

_ _ _

## 5.Transfer Learning Scholars (著名学者)

Here are some transfer learning scholars and labs.

**全部列表以及代表工作性见[这里](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)** 

Please note that this list is far not complete. A full list can be seen in [here](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md). Transfer learning is an active field. *If you are aware of some scholars, please add them here.*

_ _ _

## 6.Transfer Learning Thesis (硕博士论文)

Here are some popular thesis on transfer learning.

[这里](https://pan.baidu.com/share/init?surl=iuzZhHdumrD64-yx_VAybA), 提取码：txyz。

- - -

## 7.Datasets and Benchmarks (数据集与评测结果)

Please see [HERE](https://github.com/jindongwang/transferlearning/blob/master/data) for the popular transfer learning **datasets and benchmark** results.

[这里](https://github.com/jindongwang/transferlearning/blob/master/data)整理了常用的公开数据集和一些已发表的文章在这些数据集上的实验结果。

- - -

## 8.Transfer Learning Challenges (迁移学习比赛)

- [Visual Domain Adaptation Challenge (VisDA)](http://ai.bu.edu/visda-2018/)

- - -

## Journals and Conferences

See [here](https://github.com/jindongwang/transferlearning/blob/master/doc/venues.md) for a full list of related journals and conferences.

- - -

## Applications (迁移学习应用)

- [Computer vision](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#computer-vision)
- [Medical and healthcare](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#medical-and-healthcare)
- [Natural language processing](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#natural-language-processing)
- [Time series](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#time-series)
- [Speech](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#speech)
- [Multimedia](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#multimedia)
- [Recommendation](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#recommendation)
- [Human activity recognition](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#human-activity-recognition)
- [Autonomous driving](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#autonomous-driving)
- [Others](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md#others)

See [HERE](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md) for transfer learning applications.

迁移学习应用请见[这里](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md)。

- - -

## Other Resources (其他资源)

- Call for papers:
  - [Advances in Transfer Learning: Theory, Algorithms, and Applications](https://www.frontiersin.org/research-topics/21133/advances-in-transfer-learning-theory-algorithms-and-applications), DDL: October 2021

- Related projects:
  - Salad: [A semi-supervised domain adaptation library](https://domainadaptation.org)

- - -

## Contributing (欢迎参与贡献)

If you are interested in contributing, please refer to [HERE](https://github.com/jindongwang/transferlearning/blob/master/CONTRIBUTING.md) for instructions in contribution.

- - -

### Copyright notice

> ***[Notes]This Github repo can be used by following the corresponding licenses. I want to emphasis that it may contain some PDFs or thesis, which were downloaded by me and can only be used for academic purposes. The copyrights of these materials are owned by corresponding publishers or organizations. All this are for better academic research. If any of the authors or publishers have concerns, please contact me to delete or replace them.***

[contributors-shield]: https://img.shields.io/github/contributors/jindongwang/transferlearning.svg?style=for-the-badge
[contributors-url]: https://github.com/jindongwang/transferlearning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/jindongwang/transferlearning.svg?style=for-the-badge
[forks-url]: https://github.com/jindongwang/transferlearning/network/members
[stars-shield]: https://img.shields.io/github/stars/jindongwang/transferlearning.svg?style=for-the-badge
[stars-url]: https://github.com/jindongwang/transferlearning/stargazers
[issues-shield]: https://img.shields.io/github/issues/jindongwang/transferlearning.svg?style=for-the-badge
[issues-url]: https://github.com/jindongwang/transferlearning/issues
[license-shield]: https://img.shields.io/github/license/jindongwang/transferlearning.svg?style=for-the-badge
[license-url]: https://github.com/jindongwang/transferlearning/blob/main/LICENSE.txt


---

# AWESOME DEEP LEARNING

**Source:** https://github.com/ChristosChristofidis/awesome-deep-learning
**Stars:** 23,000+
**Resources:** 610+

## Content

﻿# Awesome Deep Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

## Table of Contents

* **[Books](#books)**

* **[Courses](#courses)**  

* **[Videos and Lectures](#videos-and-lectures)**  

* **[Papers](#papers)**  

* **[Tutorials](#tutorials)**  

* **[Researchers](#researchers)**  

* **[Websites](#websites)**  

* **[Datasets](#datasets)**

* **[Conferences](#Conferences)**

* **[Frameworks](#frameworks)**  

* **[Tools](#tools)**  

* **[Miscellaneous](#miscellaneous)**  

* **[Contributing](#contributing)**  


### Books

1.  [Deep Learning](http://www.deeplearningbook.org/) by Yoshua Bengio, Ian Goodfellow and Aaron Courville  (05/07/2015)
2.  [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by  Michael Nielsen (Dec 2014)
3.  [Deep Learning](http://research.microsoft.com/pubs/209355/DeepLearning-NowPublishing-Vol7-SIG-039.pdf) by Microsoft Research (2013)
4.  [Deep Learning Tutorial](http://deeplearning.net/tutorial/deeplearning.pdf) by LISA lab, University of Montreal (Jan 6 2015)
5.  [neuraltalk](https://github.com/karpathy/neuraltalk) by Andrej Karpathy : numpy-based RNN/LSTM implementation
6.  [An introduction to genetic algorithms](http://www.boente.eti.br/fuzzy/ebook-fuzzy-mitchell.pdf)
7.  [Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)
8.  [Deep Learning in Neural Networks: An Overview](http://arxiv.org/pdf/1404.7828v4.pdf)
9.  [Artificial intelligence and machine learning: Topic wise explanation](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/)
10. [Grokking Deep Learning for Computer Vision](https://www.manning.com/books/grokking-deep-learning-for-computer-vision)
11. [Dive into Deep Learning](https://d2l.ai/) - numpy based interactive Deep Learning book
12. [Practical Deep Learning for Cloud, Mobile, and Edge](https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/) - A book for optimization techniques during production.
13. [Math and Architectures of Deep Learning](https://www.manning.com/books/math-and-architectures-of-deep-learning) - by Krishnendu Chaudhury
14. [TensorFlow 2.0 in Action](https://www.manning.com/books/tensorflow-in-action) - by Thushan Ganegedara
15. [Deep Learning for Natural Language Processing](https://www.manning.com/books/deep-learning-for-natural-language-processing) - by Stephan Raaijmakers
16. [Deep Learning Patterns and Practices](https://www.manning.com/books/deep-learning-patterns-and-practices) - by Andrew Ferlitsch
17. [Inside Deep Learning](https://www.manning.com/books/inside-deep-learning) - by Edward Raff
18. [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition) - by François Chollet
19. [Evolutionary Deep Learning](https://www.manning.com/books/evolutionary-deep-learning) - by Micheal Lanham
20. [Engineering Deep Learning Platforms](https://www.manning.com/books/engineering-deep-learning-platforms) - by Chi Wang and Donald Szeto
21. [Deep Learning with R, Second Edition](https://www.manning.com/books/deep-learning-with-r-second-edition) - by François Chollet with Tomasz Kalinowski and J. J. Allaire
22. [Regularization in Deep Learning](https://www.manning.com/books/regularization-in-deep-learning) - by Liu Peng
23. [Jax in Action](https://www.manning.com/books/jax-in-action) - by Grigory Sapunov
24. [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.knowledgeisle.com/wp-content/uploads/2019/12/2-Aur%C3%A9lien-G%C3%A9ron-Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow_-Concepts-Tools-and-Techniques-to-Build-Intelligent-Systems-O%E2%80%99Reilly-Media-2019.pdf) by Aurélien Géron  | Oct 15, 2019

### Courses

1.  [Machine Learning - Stanford](https://class.coursera.org/ml-005) by Andrew Ng in Coursera (2010-2014)
2.  [Machine Learning - Caltech](http://work.caltech.edu/lectures.html) by Yaser Abu-Mostafa (2012-2014)
3.  [Machine Learning - Carnegie Mellon](http://www.cs.cmu.edu/~tom/10701_sp11/lectures.shtml) by Tom Mitchell (Spring 2011)
2.  [Neural Networks for Machine Learning](https://class.coursera.org/neuralnets-2012-001) by Geoffrey Hinton in Coursera (2012)
3.  [Neural networks class](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH) by Hugo Larochelle from Université de Sherbrooke (2013)
4.  [Deep Learning Course](http://cilvr.cs.nyu.edu/doku.php?id=deeplearning:slides:start) by CILVR lab @ NYU (2014)
5.  [A.I - Berkeley](https://courses.edx.org/courses/BerkeleyX/CS188x_1/1T2013/courseware/) by Dan Klein and Pieter Abbeel (2013)
6.  [A.I - MIT](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/lecture-videos/) by Patrick Henry Winston (2010)
7.  [Vision and learning - computers and brains](http://web.mit.edu/course/other/i2course/www/vision_and_learning_fall_2013.html) by Shimon Ullman, Tomaso Poggio, Ethan Meyers @ MIT (2013)
9.  [Convolutional Neural Networks for Visual Recognition - Stanford](http://vision.stanford.edu/teaching/cs231n/syllabus.html) by Fei-Fei Li, Andrej Karpathy (2017)
10.  [Deep Learning for Natural Language Processing - Stanford](http://cs224d.stanford.edu/)
11.  [Neural Networks - usherbrooke](http://info.usherbrooke.ca/hlarochelle/neural_networks/content.html)
12.  [Machine Learning - Oxford](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/) (2014-2015)
13.  [Deep Learning - Nvidia](https://developer.nvidia.com/deep-learning-courses) (2015)
14.  [Graduate Summer School: Deep Learning, Feature Learning](https://www.youtube.com/playlist?list=PLHyI3Fbmv0SdzMHAy0aN59oYnLy5vyyTA) by Geoffrey Hinton, Yoshua Bengio, Yann LeCun, Andrew Ng, Nando de Freitas and several others @ IPAM, UCLA (2012)
15.  [Deep Learning - Udacity/Google](https://www.udacity.com/course/deep-learning--ud730) by Vincent Vanhoucke and Arpan Chakraborty (2016)
16.  [Deep Learning - UWaterloo](https://www.youtube.com/playlist?list=PLehuLRPyt1Hyi78UOkMPWCGRxGcA9NVOE) by Prof. Ali Ghodsi at University of Waterloo (2015)
17.  [Statistical Machine Learning - CMU](https://www.youtube.com/watch?v=azaLcvuql_g&list=PLjbUi5mgii6BWEUZf7He6nowWvGne_Y8r) by Prof. Larry Wasserman
18.  [Deep Learning Course](https://www.college-de-france.fr/site/en-yann-lecun/course-2015-2016.htm) by Yann LeCun (2016)
19. [Designing, Visualizing and Understanding Deep Neural Networks-UC Berkeley](https://www.youtube.com/playlist?list=PLkFD6_40KJIxopmdJF_CLNqG3QuDFHQUm)
20. [UVA Deep Learning Course](http://uvadlc.github.io) MSc in Artificial Intelligence for the University of Amsterdam.
21. [MIT 6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/)
22. [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/)
23. [Berkeley CS 294: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/)
24. [Keras in Motion video course](https://www.manning.com/livevideo/keras-in-motion)
25. [Practical Deep Learning For Coders](http://course.fast.ai/) by Jeremy Howard - Fast.ai
26. [Introduction to Deep Learning](http://deeplearning.cs.cmu.edu/) by Prof. Bhiksha Raj (2017)
27. [AI for Everyone](https://www.deeplearning.ai/ai-for-everyone/) by Andrew Ng (2019)
28. [MIT Intro to Deep Learning 7 day bootcamp](https://introtodeeplearning.com) - A seven day bootcamp designed in MIT to introduce deep learning methods and applications (2019)
29. [Deep Blueberry: Deep Learning](https://mithi.github.io/deep-blueberry) - A free five-weekend plan to self-learners to learn the basics of deep-learning architectures like CNNs, LSTMs, RNNs, VAEs, GANs, DQN, A3C and more (2019)
30. [Spinning Up in Deep Reinforcement Learning](https://spinningup.openai.com/) - A free deep reinforcement learning course by OpenAI (2019)
31. [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning) - Breaking into AI with the best course from Andrew NG.
32. [Deep Learning - UC Berkeley | STAT-157](https://www.youtube.com/playlist?list=PLZSO_6-bSqHQHBCoGaObUljoXAyyqhpFW) by Alex Smola and Mu Li (2019)
33. [Machine Learning for Mere Mortals video course](https://www.manning.com/livevideo/machine-learning-for-mere-mortals) by Nick Chase
34. [Machine Learning Crash Course with TensorFlow APIs](https://developers.google.com/machine-learning/crash-course/) -Google AI
35. [Deep Learning from the Foundations](https://course.fast.ai/part2) Jeremy Howard - Fast.ai
36. [Deep Reinforcement Learning (nanodegree) - Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) a 3-6 month Udacity nanodegree, spanning multiple courses (2018)
37. [Grokking Deep Learning in Motion](https://www.manning.com/livevideo/grokking-deep-learning-in-motion) by Beau Carnes (2018)
38. [Face Detection with Computer Vision and Deep Learning](https://www.udemy.com/share/1000gAA0QdcV9aQng=/) by Hakan Cebeci
39. [Deep Learning Online Course list at Classpert](https://classpert.com/deep-learning) List of Deep Learning online courses (some are free) from Classpert Online Course Search
40. [AWS Machine Learning](https://aws.training/machinelearning) Machine Learning and Deep Learning Courses from Amazon's Machine Learning university
41. [Intro to Deep Learning with PyTorch](https://www.udacity.com/course/deep-learning-pytorch--ud188) - A great introductory course on Deep Learning by Udacity and Facebook AI
42. [Deep Learning by Kaggle](https://www.kaggle.com/learn/deep-learning) - Kaggle's  free course on Deep Learning
43. [Yann LeCun’s Deep Learning Course at CDS](https://cds.nyu.edu/deep-learning/) - DS-GA 1008 · SPRING 2021 
44. [Neural Networks and Deep Learning](https://webcms3.cse.unsw.edu.au/COMP9444/19T3/) - COMP9444 19T3
45. [Deep Learning A.I.Shelf](http://aishelf.org/category/ia/deep-learning/)

### Videos and Lectures

1.  [How To Create A Mind](https://www.youtube.com/watch?v=RIkxVci-R4k) By Ray Kurzweil
2.  [Deep Learning, Self-Taught Learning and Unsupervised Feature Learning](https://www.youtube.com/watch?v=n1ViNeWhC24) By Andrew Ng
3.  [Recent Developments in Deep Learning](https://www.youtube.com/watch?v=vShMxxqtDDs&amp;index=3&amp;list=PL78U8qQHXgrhP9aZraxTT5-X1RccTcUYT) By Geoff Hinton
4.  [The Unreasonable Effectiveness of Deep Learning](https://www.youtube.com/watch?v=sc-KbuZqGkI) by Yann LeCun
5.  [Deep Learning of Representations](https://www.youtube.com/watch?v=4xsVFLnHC_0) by Yoshua bengio
6.  [Principles of Hierarchical Temporal Memory](https://www.youtube.com/watch?v=6ufPpZDmPKA) by Jeff Hawkins
7.  [Machine Learning Discussion Group - Deep Learning w/ Stanford AI Lab](https://www.youtube.com/watch?v=2QJi0ArLq7s&amp;list=PL78U8qQHXgrhP9aZraxTT5-X1RccTcUYT) by Adam Coates
8.  [Making Sense of the World with Deep Learning](http://vimeo.com/80821560) By Adam Coates
9.  [Demystifying Unsupervised Feature Learning ](https://www.youtube.com/watch?v=wZfVBwOO0-k) By Adam Coates
10.  [Visual Perception with Deep Learning](https://www.youtube.com/watch?v=3boKlkPBckA) By Yann LeCun
11.  [The Next Generation of Neural Networks](https://www.youtube.com/watch?v=AyzOUbkUf3M) By Geoffrey Hinton at GoogleTechTalks
12.  [The wonderful and terrifying implications of computers that can learn](http://www.ted.com/talks/jeremy_howard_the_wonderful_and_terrifying_implications_of_computers_that_can_learn) By Jeremy Howard at TEDxBrussels
13.  [Unsupervised Deep Learning - Stanford](http://web.stanford.edu/class/cs294a/handouts.html) by Andrew Ng in Stanford (2011)
14.  [Natural Language Processing](http://web.stanford.edu/class/cs224n/handouts/) By Chris Manning in Stanford
15.  [A beginners Guide to Deep Neural Networks](http://googleresearch.blogspot.com/2015/09/a-beginners-guide-to-deep-neural.html) By Natalie Hammel and Lorraine Yurshansky
16.  [Deep Learning: Intelligence from Big Data](https://www.youtube.com/watch?v=czLI3oLDe8M) by Steve Jurvetson (and panel) at VLAB in Stanford.
17. [Introduction to Artificial Neural Networks and Deep Learning](https://www.youtube.com/watch?v=FoO8qDB8gUU) by Leo Isikdogan at Motorola Mobility HQ
18. [NIPS 2016 lecture and workshop videos](https://nips.cc/Conferences/2016/Schedule) - NIPS 2016
19. [Deep Learning Crash Course](https://www.youtube.com/watch?v=oS5fz_mHVz0&list=PLWKotBjTDoLj3rXBL-nEIPRN9V3a9Cx07): a series of mini-lectures by Leo Isikdogan on YouTube (2018)
20. [Deep Learning Crash Course](https://www.manning.com/livevideo/deep-learning-crash-course) By Oliver Zeigermann
21. [Deep Learning with R in Motion](https://www.manning.com/livevideo/deep-learning-with-r-in-motion): a live video course that teaches how to apply deep learning to text and images using the powerful Keras library and its R language interface.
22. [Medical Imaging with Deep Learning Tutorial](https://www.youtube.com/playlist?list=PLheiZMDg_8ufxEx9cNVcOYXsT3BppJP4b): This tutorial is styled as a graduate lecture about medical imaging with deep learning. This will cover the background of popular medical image domains (chest X-ray and histology) as well as methods to tackle multi-modality/view, segmentation, and counting tasks.
23. [Deepmind x UCL Deeplearning](https://www.youtube.com/playlist?list=PLqYmG7hTraZCDxZ44o4p3N5Anz3lLRVZF): 2020 version 
24. [Deepmind x UCL Reinforcement Learning](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb): Deep Reinforcement Learning
25. [CMU 11-785 Intro to Deep learning Spring 2020](https://www.youtube.com/playlist?list=PLp-0K3kfddPzCnS4CqKphh-zT3aDwybDe) Course: 11-785, Intro to Deep Learning by Bhiksha Raj 
26. [Machine Learning CS 229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) : End part focuses on deep learning By Andrew Ng
27. [What is Neural Structured Learning by Andrew Ferlitsch](https://youtu.be/LXWSE_9gHd0)
28. [Deep Learning Design Patterns by Andrew Ferlitsch](https://youtu.be/_DaviS6K0Vc)
29. [Architecture of a Modern CNN: the design pattern approach by Andrew Ferlitsch](https://youtu.be/QCGSS3kyGo0)
30. [Metaparameters in a CNN by Andrew Ferlitsch](https://youtu.be/K1PLeggQ33I)
31. [Multi-task CNN: a real-world example by Andrew Ferlitsch](https://youtu.be/dH2nuI-1-qM)
32. [A friendly introduction to deep reinforcement learning by Luis Serrano](https://youtu.be/1FyAh07jh0o)
33. [What are GANs and how do they work? by Edward Raff](https://youtu.be/f6ivp84qFUc)
34. [Coding a basic WGAN in PyTorch by Edward Raff](https://youtu.be/7VRdaqMDalQ)
35. [Training a Reinforcement Learning Agent by Miguel Morales](https://youtu.be/8TMT-gHlj_Q)
36. [Understand what is Deep Learning](https://www.scaler.com/topics/what-is-deep-learning/)

### Papers
*You can also find the most cited deep learning papers from [here](https://github.com/terryum/awesome-deep-learning-papers)*

1.  [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
2.  [Using Very Deep Autoencoders for Content Based Image Retrieval](http://www.cs.toronto.edu/~hinton/absps/esann-deep-final.pdf)
3.  [Learning Deep Architectures for AI](http://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf)
4.  [CMU’s list of papers](http://deeplearning.cs.cmu.edu/)
5.  [Neural Networks for Named Entity Recognition](http://nlp.stanford.edu/~socherr/pa4_ner.pdf) [zip](http://nlp.stanford.edu/~socherr/pa4-ner.zip)
6. [Training tricks by YB](http://www.iro.umontreal.ca/~bengioy/papers/YB-tricks.pdf)
7. [Geoff Hinton's reading list (all papers)](http://www.cs.toronto.edu/~hinton/deeprefs.html)
8. [Supervised Sequence Labelling with Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
9.  [Statistical Language Models based on Neural Networks](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
10.  [Training Recurrent Neural Networks](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
11.  [Recursive Deep Learning for Natural Language Processing and Computer Vision](http://nlp.stanford.edu/~socherr/thesis.pdf)
12.  [Bi-directional RNN](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)
13.  [LSTM](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)
14.  [GRU - Gated Recurrent Unit](http://arxiv.org/pdf/1406.1078v3.pdf)
15.  [GFRNN](http://arxiv.org/pdf/1502.02367v3.pdf) [.](http://jmlr.org/proceedings/papers/v37/chung15.pdf) [.](http://jmlr.org/proceedings/papers/v37/chung15-supp.pdf)
16.  [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069v1.pdf)
17.  [A Critical Review of Recurrent Neural Networks for Sequence Learning](http://arxiv.org/pdf/1506.00019v1.pdf)
18.  [Visualizing and Understanding Recurrent Networks](http://arxiv.org/pdf/1506.02078v1.pdf)
19.  [Wojciech Zaremba, Ilya Sutskever, An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
20.  [Recurrent Neural Network based Language Model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
21.  [Extensions of Recurrent Neural Network Language Model](http://www.fit.vutbr.cz/research/groups/speech/publi/2011/mikolov_icassp2011_5528.pdf)
22.  [Recurrent Neural Network based Language Modeling in Meeting Recognition](http://www.fit.vutbr.cz/~imikolov/rnnlm/ApplicationOfRNNinMeetingRecognition_IS2011.pdf)
23.  [Deep Neural Networks for Acoustic Modeling in Speech Recognition](http://cs224d.stanford.edu/papers/maas_paper.pdf)
24.  [Speech Recognition with Deep Recurrent Neural Networks](http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)
25.  [Reinforcement Learning Neural Turing Machines](http://arxiv.org/pdf/1505.00521v1)
26.  [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/pdf/1406.1078v3.pdf)
27. [Google - Sequence to Sequence  Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
28. [Memory Networks](http://arxiv.org/pdf/1410.3916v10)
29. [Policy Learning with Continuous Memory States for Partially Observed Robotic Control](http://arxiv.org/pdf/1507.01273v1)
30. [Microsoft - Jointly Modeling Embedding and Translation to Bridge Video and Language](http://arxiv.org/pdf/1505.01861v1.pdf)
31. [Neural Turing Machines](http://arxiv.org/pdf/1410.5401v2.pdf)
32. [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://arxiv.org/pdf/1506.07285v1.pdf)
33. [Mastering the Game of Go with Deep Neural Networks and Tree Search](http://www.nature.com/nature/journal/v529/n7587/pdf/nature16961.pdf)
34. [Batch Normalization](https://arxiv.org/abs/1502.03167)
35. [Residual Learning](https://arxiv.org/pdf/1512.03385v1.pdf)
36. [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf)
37. [Berkeley AI Research (BAIR) Laboratory](https://arxiv.org/pdf/1611.07004v1.pdf)
38. [MobileNets by Google](https://arxiv.org/abs/1704.04861)
39. [Cross Audio-Visual Recognition in the Wild Using Deep Learning](https://arxiv.org/abs/1706.05739)
40. [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
41. [Matrix Capsules With Em Routing](https://openreview.net/pdf?id=HJWLfGWRb)
42. [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
43. [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661v1.pdf)
44. [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
45. [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
46. [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
47. [Unsupervised Translation of Programming Languages](https://arxiv.org/pdf/2006.03511.pdf)
48. [Matching Networks for One Shot Learning](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)
49. [VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/pdf/2106.13112.pdf)
50. [ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)
51. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://proceedings.mlr.press/v37/ioffe15.pdf)
52. [DeepFaceDrawing: Deep Generation of Face Images from Sketches](http://geometrylearning.com/paper/DeepFaceDrawing.pdf?fbclid=IwAR0colWFHPGBCB1APZq9JVsWeWtmeZd9oCTNQvR52T5PRUJP_dLOwB8pt0I)

### Tutorials

1.  [UFLDL Tutorial 1](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial)
2.  [UFLDL Tutorial 2](http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/)
3.  [Deep Learning for NLP (without Magic)](http://www.socher.org/index.php/DeepLearningTutorial/DeepLearningTutorial)
4.  [A Deep Learning Tutorial: From Perceptrons to Deep Networks](http://www.toptal.com/machine-learning/an-introduction-to-deep-learning-from-perceptrons-to-deep-networks)
5.  [Deep Learning from the Bottom up](http://www.metacademy.org/roadmaps/rgrosse/deep_learning)
6.  [Theano Tutorial](http://deeplearning.net/tutorial/deeplearning.pdf)
7.  [Neural Networks for Matlab](http://uk.mathworks.com/help/pdf_doc/nnet/nnet_ug.pdf)
8.  [Using convolutional neural nets to detect facial keypoints tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)
9.  [Torch7 Tutorials](https://github.com/clementfarabet/ipam-tutorials/tree/master/th_tutorials)
10.  [The Best Machine Learning Tutorials On The Web](https://github.com/josephmisiti/machine-learning-module)
11. [VGG Convolutional Neural Networks Practical](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html)
12. [TensorFlow tutorials](https://github.com/nlintz/TensorFlow-Tutorials)
13. [More TensorFlow tutorials](https://github.com/pkmital/tensorflow_tutorials)
13. [TensorFlow Python Notebooks](https://github.com/aymericdamien/TensorFlow-Examples)
14. [Keras and Lasagne Deep Learning Tutorials](https://github.com/Vict0rSch/deep_learning)
15. [Classification on raw time series in TensorFlow with a LSTM RNN](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)
16. [Using convolutional neural nets to detect facial keypoints tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)
17. [TensorFlow-World](https://github.com/astorfi/TensorFlow-World)
18. [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)
19. [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning)
20. [Deep Learning for Search](https://www.manning.com/books/deep-learning-for-search)
21. [Keras Tutorial: Content Based Image Retrieval Using a Convolutional Denoising Autoencoder](https://medium.com/sicara/keras-tutorial-content-based-image-retrieval-convolutional-denoising-autoencoder-dc91450cc511)
22. [Pytorch Tutorial by Yunjey Choi](https://github.com/yunjey/pytorch-tutorial)
23. [Understanding deep Convolutional Neural Networks with a practical use-case in Tensorflow and Keras](https://ahmedbesbes.com/understanding-deep-convolutional-neural-networks-with-a-practical-use-case-in-tensorflow-and-keras.html)
24. [Overview and benchmark of traditional and deep learning models in text classification](https://ahmedbesbes.com/overview-and-benchmark-of-traditional-and-deep-learning-models-in-text-classification.html)
25. [Hardware for AI: Understanding computer hardware & build your own computer](https://github.com/MelAbgrall/HardwareforAI)
26. [Programming Community Curated Resources](https://hackr.io/tutorials/learn-artificial-intelligence-ai)
27. [The Illustrated Self-Supervised Learning](https://amitness.com/2020/02/illustrated-self-supervised-learning/)
28. [Visual Paper Summary: ALBERT (A Lite BERT)](https://amitness.com/2020/02/albert-visual-summary/)
28. [Semi-Supervised Deep Learning with GANs for Melanoma Detection](https://www.manning.com/liveproject/semi-supervised-deep-learning-with-gans-for-melanoma-detection/)
29. [Named Entity Recognition using Reformers](https://github.com/SauravMaheshkar/Trax-Examples/blob/main/NLP/NER%20using%20Reformer.ipynb)
30. [Deep N-Gram Models on Shakespeare’s works](https://github.com/SauravMaheshkar/Trax-Examples/blob/main/NLP/Deep%20N-Gram.ipynb)
31. [Wide Residual Networks](https://github.com/SauravMaheshkar/Trax-Examples/blob/main/vision/illustrated-wideresnet.ipynb)
32. [Fashion MNIST using Flax](https://github.com/SauravMaheshkar/Flax-Examples)
33. [Fake News Classification (with streamlit deployment)](https://github.com/SauravMaheshkar/Fake-News-Classification)
34. [Regression Analysis for Primary Biliary Cirrhosis](https://github.com/SauravMaheshkar/CoxPH-Model-for-Primary-Biliary-Cirrhosis)
35. [Cross Matching Methods for Astronomical Catalogs](https://github.com/SauravMaheshkar/Cross-Matching-Methods-for-Astronomical-Catalogs)
36. [Named Entity Recognition using BiDirectional LSTMs](https://github.com/SauravMaheshkar/Named-Entity-Recognition-)
37. [Image Recognition App using Tflite and Flutter](https://github.com/SauravMaheshkar/Flutter_Image-Recognition)

## Researchers

1. [Aaron Courville](http://aaroncourville.wordpress.com)
2. [Abdel-rahman Mohamed](http://www.cs.toronto.edu/~asamir/)
3. [Adam Coates](http://cs.stanford.edu/~acoates/)
4. [Alex Acero](http://research.microsoft.com/en-us/people/alexac/)
5. [ Alex Krizhevsky ](http://www.cs.utoronto.ca/~kriz/index.html)
6. [ Alexander Ilin ](http://users.ics.aalto.fi/alexilin/)
7. [ Amos Storkey ](http://homepages.inf.ed.ac.uk/amos/)
8. [ Andrej Karpathy ](https://karpathy.ai/)
9. [ Andrew M. Saxe ](http://www.stanford.edu/~asaxe/)
10. [ Andrew Ng ](http://www.cs.stanford.edu/people/ang/)
11. [ Andrew W. Senior ](http://research.google.com/pubs/author37792.html)
12. [ Andriy Mnih ](http://www.gatsby.ucl.ac.uk/~amnih/)
13. [ Ayse Naz Erkan ](http://www.cs.nyu.edu/~naz/)
14. [ Benjamin Schrauwen ](http://reslab.elis.ugent.be/benjamin)
15. [ Bernardete Ribeiro ](https://www.cisuc.uc.pt/people/show/2020)
16. [ Bo David Chen ](http://vision.caltech.edu/~bchen3/Site/Bo_David_Chen.html)
17. [ Boureau Y-Lan ](http://cs.nyu.edu/~ylan/)
18. [ Brian Kingsbury ](http://researcher.watson.ibm.com/researcher/view.php?person=us-bedk)
19. [ Christopher Manning ](http://nlp.stanford.edu/~manning/)
20. [ Clement Farabet ](http://www.clement.farabet.net/)
21. [ Dan Claudiu Cireșan ](http://www.idsia.ch/~ciresan/)
22. [ David Reichert ](http://serre-lab.clps.brown.edu/person/david-reichert/)
23. [ Derek Rose ](http://mil.engr.utk.edu/nmil/member/5.html)
24. [ Dong Yu ](http://research.microsoft.com/en-us/people/dongyu/default.aspx)
25. [ Drausin Wulsin ](http://www.seas.upenn.edu/~wulsin/)
26. [ Erik M. Schmidt ](http://music.ece.drexel.edu/people/eschmidt)
27. [ Eugenio Culurciello ](https://engineering.purdue.edu/BME/People/viewPersonById?resource_id=71333)
28. [ Frank Seide ](http://research.microsoft.com/en-us/people/fseide/)
29. [ Galen Andrew ](http://homes.cs.washington.edu/~galen/)
30. [ Geoffrey Hinton ](http://www.cs.toronto.edu/~hinton/)
31. [ George Dahl ](http://www.cs.toronto.edu/~gdahl/)
32. [ Graham Taylor ](http://www.uoguelph.ca/~gwtaylor/)
33. [ Grégoire Montavon ](http://gregoire.montavon.name/)
34. [ Guido Francisco Montúfar ](http://personal-homepages.mis.mpg.de/montufar/)
35. [ Guillaume Desjardins ](http://brainlogging.wordpress.com/)
36. [ Hannes Schulz ](http://www.ais.uni-bonn.de/~schulz/)
37. [ Hélène Paugam-Moisy ](http://www.lri.fr/~hpaugam/)
38. [ Honglak Lee ](http://web.eecs.umich.edu/~honglak/)
39. [ Hugo Larochelle ](http://www.dmi.usherb.ca/~larocheh/index_en.html)
40. [ Ilya Sutskever ](http://www.cs.toronto.edu/~ilya/)
41. [ Itamar Arel ](http://mil.engr.utk.edu/nmil/member/2.html)
42. [ James Martens ](http://www.cs.toronto.edu/~jmartens/)
43. [ Jason Morton ](http://www.jasonmorton.com/)
44. [ Jason Weston ](http://www.thespermwhale.com/jaseweston/)
45. [ Jeff Dean ](http://research.google.com/pubs/jeff.html)
46. [ Jiquan Mgiam ](http://cs.stanford.edu/~jngiam/)
47. [ Joseph Turian ](http://www-etud.iro.umontreal.ca/~turian/)
48. [ Joshua Matthew Susskind ](http://aclab.ca/users/josh/index.html)
49. [ Jürgen Schmidhuber ](http://www.idsia.ch/~juergen/)
50. [ Justin A. Blanco ](https://sites.google.com/site/blancousna/)
51. [ Koray Kavukcuoglu ](http://koray.kavukcuoglu.org/)
52. [ KyungHyun Cho ](http://users.ics.aalto.fi/kcho/)
53. [ Li Deng ](http://research.microsoft.com/en-us/people/deng/)
54. [ Lucas Theis ](http://www.kyb.tuebingen.mpg.de/nc/employee/details/lucas.html)
55. [ Ludovic Arnold ](http://ludovicarnold.altervista.org/home/)
56. [ Marc'Aurelio Ranzato ](http://www.cs.nyu.edu/~ranzato/)
57. [ Martin Längkvist ](http://aass.oru.se/~mlt/)
58. [ Misha Denil ](http://mdenil.com/)
59. [ Mohammad Norouzi ](http://www.cs.toronto.edu/~norouzi/)
60. [ Nando de Freitas ](http://www.cs.ubc.ca/~nando/)
61. [ Navdeep Jaitly ](http://www.cs.utoronto.ca/~ndjaitly/)
62. [ Nicolas Le Roux ](http://nicolas.le-roux.name/)
63. [ Nitish Srivastava ](http://www.cs.toronto.edu/~nitish/)
64. [ Noel Lopes ](https://www.cisuc.uc.pt/people/show/2028)
65. [ Oriol Vinyals ](http://www.cs.berkeley.edu/~vinyals/)
66. [ Pascal Vincent ](http://www.iro.umontreal.ca/~vincentp)
67. [ Patrick Nguyen ](https://sites.google.com/site/drpngx/)
68. [ Pedro Domingos ](http://homes.cs.washington.edu/~pedrod/)
69. [ Peggy Series ](http://homepages.inf.ed.ac.uk/pseries/)
70. [ Pierre Sermanet ](http://cs.nyu.edu/~sermanet)
71. [ Piotr Mirowski ](http://www.cs.nyu.edu/~mirowski/)
72. [ Quoc V. Le ](http://ai.stanford.edu/~quocle/)
73. [ Reinhold Scherer ](http://bci.tugraz.at/scherer/)
74. [ Richard Socher ](http://www.socher.org/)
75. [ Rob Fergus ](http://cs.nyu.edu/~fergus/pmwiki/pmwiki.php)
76. [ Robert Coop ](http://mil.engr.utk.edu/nmil/member/19.html)
77. [ Robert Gens ](http://homes.cs.washington.edu/~rcg/)
78. [ Roger Grosse ](http://people.csail.mit.edu/rgrosse/)
79. [ Ronan Collobert ](http://ronan.collobert.com/)
80. [ Ruslan Salakhutdinov ](http://www.utstat.toronto.edu/~rsalakhu/)
81. [ Sebastian Gerwinn ](http://www.kyb.tuebingen.mpg.de/nc/employee/details/sgerwinn.html)
82. [ Stéphane Mallat ](http://www.cmap.polytechnique.fr/~mallat/)
83. [ Sven Behnke ](http://www.ais.uni-bonn.de/behnke/)
84. [ Tapani Raiko ](http://users.ics.aalto.fi/praiko/)
85. [ Tara Sainath ](https://sites.google.com/site/tsainath/)
86. [ Tijmen Tieleman ](http://www.cs.toronto.edu/~tijmen/)
87. [ Tom Karnowski ](http://mil.engr.utk.edu/nmil/member/36.html)
88. [ Tomáš Mikolov ](https://research.facebook.com/tomas-mikolov)
89. [ Ueli Meier ](http://www.idsia.ch/~meier/)
90. [ Vincent Vanhoucke ](http://vincent.vanhoucke.com)
91. [ Volodymyr Mnih ](http://www.cs.toronto.edu/~vmnih/)
92. [ Yann LeCun ](http://yann.lecun.com/)
93. [ Yichuan Tang ](http://www.cs.toronto.edu/~tang/)
94. [ Yoshua Bengio ](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html)
95. [ Yotaro Kubo ](http://yota.ro/)
96. [ Youzhi (Will) Zou ](http://ai.stanford.edu/~wzou)
97. [ Fei-Fei Li ](http://vision.stanford.edu/feifeili)
98. [ Ian Goodfellow ](https://research.google.com/pubs/105214.html)
99. [ Robert Laganière ](http://www.site.uottawa.ca/~laganier/)
100. [Merve Ayyüce Kızrak](http://www.ayyucekizrak.com/)


### Websites

1.  [deeplearning.net](http://deeplearning.net/)
2.  [deeplearning.stanford.edu](http://deeplearning.stanford.edu/)
3.  [nlp.stanford.edu](http://nlp.stanford.edu/)
4.  [ai-junkie.com](http://www.ai-junkie.com/ann/evolved/nnt1.html)
5.  [cs.brown.edu/research/ai](http://cs.brown.edu/research/ai/)
6.  [eecs.umich.edu/ai](http://www.eecs.umich.edu/ai/)
7.  [cs.utexas.edu/users/ai-lab](http://www.cs.utexas.edu/users/ai-lab/)
8.  [cs.washington.edu/research/ai](http://www.cs.washington.edu/research/ai/)
9.  [aiai.ed.ac.uk](http://www.aiai.ed.ac.uk/)
10.  [www-aig.jpl.nasa.gov](http://www-aig.jpl.nasa.gov/)
11.  [csail.mit.edu](http://www.csail.mit.edu/)
12.  [cgi.cse.unsw.edu.au/~aishare](http://cgi.cse.unsw.edu.au/~aishare/)
13.  [cs.rochester.edu/research/ai](http://www.cs.rochester.edu/research/ai/)
14.  [ai.sri.com](http://www.ai.sri.com/)
15.  [isi.edu/AI/isd.htm](http://www.isi.edu/AI/isd.htm)
16.  [nrl.navy.mil/itd/aic](http://www.nrl.navy.mil/itd/aic/)
17.  [hips.seas.harvard.edu](http://hips.seas.harvard.edu/)
18.  [AI Weekly](http://aiweekly.co)
19.  [stat.ucla.edu](http://statistics.ucla.edu/)
20.  [deeplearning.cs.toronto.edu](http://deeplearning.cs.toronto.edu/i2t)
21.  [jeffdonahue.com/lrcn/](http://jeffdonahue.com/lrcn/)
22.  [visualqa.org](http://www.visualqa.org/)
23.  [www.mpi-inf.mpg.de/departments/computer-vision...](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/)
24.  [Deep Learning News](http://news.startup.ml/)
25.  [Machine Learning is Fun! Adam Geitgey's Blog](https://medium.com/@ageitgey/)
26.  [Guide to Machine Learning](http://yerevann.com/a-guide-to-deep-learning/)
27.  [Deep Learning for Beginners](https://spandan-madan.github.io/DeepLearningProject/)
28.  [Machine Learning Mastery blog](https://machinelearningmastery.com/blog/)
29.  [ML Compiled](https://ml-compiled.readthedocs.io/en/latest/)
30.  [Programming Community Curated Resources](https://hackr.io/tutorials/learn-artificial-intelligence-ai)
31.  [A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/)
32.  [ahmedbesbes.com](http://ahmedbesbes.com)
33.  [amitness.com](https://amitness.com/)
34.  [AI Summer](https://theaisummer.com/)
35.  [AI Hub - supported by AAAI, NeurIPS](https://aihub.org/)
36.  [CatalyzeX: Machine Learning Hub for Builders and Makers](https://www.catalyzeX.com)
37.  [The Epic Code](https://theepiccode.com/)
38.  [all AI news](https://allainews.com/)

### Datasets

1.  [MNIST](http://yann.lecun.com/exdb/mnist/) Handwritten digits
2.  [Google House Numbers](http://ufldl.stanford.edu/housenumbers/) from street view
3.  [CIFAR-10 and CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)
4.  [IMAGENET](http://www.image-net.org/)
5.  [Tiny Images](http://groups.csail.mit.edu/vision/TinyImages/) 80 Million tiny images6.  
6.  [Flickr Data](https://yahooresearch.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images) 100 Million Yahoo dataset
7.  [Berkeley Segmentation Dataset 500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
8.  [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/)
9.  [Flickr 8k](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)
10. [Flickr 30k](http://shannon.cs.illinois.edu/DenotationGraph/)
11. [Microsoft COCO](http://mscoco.org/home/)
12. [VQA](http://www.visualqa.org/)
13. [Image QA](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/)
14. [AT&T Laboratories Cambridge face database](http://www.uk.research.att.com/facedatabase.html)
15. [AVHRR Pathfinder](http://xtreme.gsfc.nasa.gov)
16. [Air Freight](http://www.anc.ed.ac.uk/~amos/afreightdata.html) - The Air Freight data set is a ray-traced image sequence along with ground truth segmentation based on textural characteristics. (455 images + GT, each 160x120 pixels). (Formats: PNG)  
17. [Amsterdam Library of Object Images](http://www.science.uva.nl/~aloi/) - ALOI is a color image collection of one-thousand small objects, recorded for scientific purposes. In order to capture the sensory variation in object recordings, we systematically varied viewing angle, illumination angle, and illumination color for each object, and additionally captured wide-baseline stereo images. We recorded over a hundred images of each object, yielding a total of 110,250 images for the collection. (Formats: png)
18. [Annotated face, hand, cardiac & meat images](http://www.imm.dtu.dk/~aam/) - Most images & annotations are supplemented by various ASM/AAM analyses using the AAM-API. (Formats: bmp,asf)
19. [Image Analysis and Computer Graphics](http://www.imm.dtu.dk/image/)  
21. [Brown University Stimuli](http://www.cog.brown.edu/~tarr/stimuli.html) - A variety of datasets including geons, objects, and "greebles". Good for testing recognition algorithms. (Formats: pict)
22. [CAVIAR video sequences of mall and public space behavior](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) - 90K video frames in 90 sequences of various human activities, with XML ground truth of detection and behavior classification (Formats: MPEG2 & JPEG)
23. [Machine Vision Unit](http://www.ipab.inf.ed.ac.uk/mvu/)
25. [CCITT Fax standard images](http://www.cs.waikato.ac.nz/~singlis/ccitt.html) - 8 images (Formats: gif)
26. [CMU CIL's Stereo Data with Ground Truth](cil-ster.html) - 3 sets of 11 images, including color tiff images with spectroradiometry (Formats: gif, tiff)
27. [CMU PIE Database](http://www.ri.cmu.edu/projects/project_418.html) - A database of 41,368 face images of 68 people captured under 13 poses, 43 illuminations conditions, and with 4 different expressions.
28. [CMU VASC Image Database](http://www.ius.cs.cmu.edu/idb/) - Images, sequences, stereo pairs (thousands of images) (Formats: Sun Rasterimage)
29. [Caltech Image Database](http://www.vision.caltech.edu/html-files/archive.html) - about 20 images - mostly top-down views of small objects and toys. (Formats: GIF)
30. [Columbia-Utrecht Reflectance and Texture Database](http://www.cs.columbia.edu/CAVE/curet/) - Texture and reflectance measurements for over 60 samples of 3D texture, observed with over 200 different combinations of viewing and illumination directions. (Formats: bmp)
31. [Computational Colour Constancy Data](http://www.cs.sfu.ca/~colour/data/index.html) - A dataset oriented towards computational color constancy, but useful for computer vision in general. It includes synthetic data, camera sensor data, and over 700 images. (Formats: tiff)
32. [Computational Vision Lab](http://www.cs.sfu.ca/~colour/)
34. [Content-based image retrieval database](http://www.cs.washington.edu/research/imagedatabase/groundtruth/) - 11 sets of color images for testing algorithms for content-based retrieval. Most sets have a description file with names of objects in each image. (Formats: jpg)
35. [Efficient Content-based Retrieval Group](http://www.cs.washington.edu/research/imagedatabase/)
37. [Densely Sampled View Spheres](http://ls7-www.cs.uni-dortmund.de/~peters/pages/research/modeladaptsys/modeladaptsys_vba_rov.html) - Densely sampled view spheres - upper half of the view sphere of two toy objects with 2500 images each. (Formats: tiff)
38. [Computer Science VII (Graphical Systems)](http://ls7-www.cs.uni-dortmund.de/)
40. [Digital Embryos](https://web-beta.archive.org/web/20011216051535/vision.psych.umn.edu/www/kersten-lab/demos/digitalembryo.html) - Digital embryos are novel objects which may be used to develop and test object recognition systems. They have an organic appearance. (Formats: various formats are available on request)
41. [Univerity of Minnesota Vision Lab](http://vision.psych.umn.edu/users/kersten//kersten-lab/kersten-lab.html) 
42. [El Salvador Atlas of Gastrointestinal VideoEndoscopy](http://www.gastrointestinalatlas.com) - Images and Videos of his-res of studies taken from Gastrointestinal Video endoscopy. (Formats: jpg, mpg, gif)
43. [FG-NET Facial Aging Database](http://sting.cycollege.ac.cy/~alanitis/fgnetaging/index.htm) - Database contains 1002 face images showing subjects at different ages. (Formats: jpg)
44. [FVC2000 Fingerprint Databases](http://bias.csr.unibo.it/fvc2000/) - FVC2000 is the First International Competition for Fingerprint Verification Algorithms. Four fingerprint databases constitute the FVC2000 benchmark (3520 fingerprints in all).
45. [Biometric Systems Lab](http://biolab.csr.unibo.it/home.asp) - University of Bologna
46. [Face and Gesture images and image sequences](http://www.fg-net.org) - Several image datasets of faces and gestures that are ground truth annotated for benchmarking
47. [German Fingerspelling Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database.html) - The database contains 35 gestures and consists of 1400 image sequences that contain gestures of 20 different persons recorded under non-uniform daylight lighting conditions. (Formats: mpg,jpg)  
48. [Language Processing and Pattern Recognition](http://www-i6.informatik.rwth-aachen.de/)
50. [Groningen Natural Image Database](http://hlab.phys.rug.nl/archive.html) - 4000+ 1536x1024 (16 bit) calibrated outdoor images (Formats: homebrew)
51. [ICG Testhouse sequence](http://www.icg.tu-graz.ac.at/~schindler/Data) -  2 turntable sequences from different viewing heights, 36 images each, resolution 1000x750, color (Formats: PPM)
52. [Institute of Computer Graphics and Vision](http://www.icg.tu-graz.ac.at)
54. [IEN Image Library](http://www.ien.it/is/vislib/) - 1000+ images, mostly outdoor sequences (Formats: raw, ppm)  
55. [INRIA's Syntim images database](http://www-rocq.inria.fr/~tarel/syntim/images.html) - 15 color image of simple objects (Formats: gif)
56. [INRIA](http://www.inria.fr/)
57. [INRIA's Syntim stereo databases](http://www-rocq.inria.fr/~tarel/syntim/paires.html) - 34 calibrated color stereo pairs (Formats: gif)
58. [Image Analysis Laboratory](http://www.ece.ncsu.edu/imaging/Archives/ImageDataBase/index.html) - Images obtained from a variety of imaging modalities -- raw CFA images, range images and a host of "medical images". (Formats: homebrew)
59. [Image Analysis Laboratory](http://www.ece.ncsu.edu/imaging)
61. [Image Database](http://www.prip.tuwien.ac.at/prip/image.html) - An image database including some textures  
62. [JAFFE Facial Expression Image Database](http://www.mis.atr.co.jp/~mlyons/jaffe.html) - The JAFFE database consists of 213 images of Japanese female subjects posing 6 basic facial expressions as well as a neutral pose. Ratings on emotion adjectives are also available, free of charge, for research purposes. (Formats: TIFF Grayscale images.)
63. [ATR Research, Kyoto, Japan](http://www.mic.atr.co.jp/)
64. [JISCT Stereo Evaluation](ftp://ftp.vislist.com/IMAGERY/JISCT/) - 44 image pairs. These data have been used in an evaluation of stereo analysis, as described in the April 1993 ARPA Image Understanding Workshop paper ``The JISCT Stereo Evaluation'' by R.C.Bolles, H.H.Baker, and M.J.Hannah, 263--274 (Formats: SSI)
65. [MIT Vision Texture](https://vismod.media.mit.edu/vismod/imagery/VisionTexture/vistex.html) - Image archive (100+ images) (Formats: ppm)
66. [MIT face images and more](ftp://whitechapel.media.mit.edu/pub/images) - hundreds of images (Formats: homebrew)
67. [Machine Vision](http://vision.cse.psu.edu/book/testbed/images/) - Images from the textbook by Jain, Kasturi, Schunck (20+ images) (Formats: GIF TIFF)
68. [Mammography Image Databases](http://marathon.csee.usf.edu/Mammography/Database.html) - 100 or more images of mammograms with ground truth. Additional images available by request, and links to several other mammography databases are provided. (Formats: homebrew)
69. [ftp://ftp.cps.msu.edu/pub/prip](ftp://ftp.cps.msu.edu/pub/prip) - many images (Formats: unknown)
70. [Middlebury Stereo Data Sets with Ground Truth](http://www.middlebury.edu/stereo/data.html) - Six multi-frame stereo data sets of scenes containing planar regions. Each data set contains 9 color images and subpixel-accuracy ground-truth data. (Formats: ppm)
71. [Middlebury Stereo Vision Research Page](http://www.middlebury.edu/stereo) - Middlebury College
72. [Modis Airborne simulator, Gallery and data set](http://ltpwww.gsfc.nasa.gov/MODIS/MAS/) - High Altitude Imagery from around the world for environmental modeling in support of NASA EOS program (Formats: JPG and HDF)
73. [NIST Fingerprint and handwriting](ftp://sequoyah.ncsl.nist.gov/pub/databases/data) - datasets - thousands of images (Formats: unknown)
74. [NIST Fingerprint data](ftp://ftp.cs.columbia.edu/jpeg/other/uuencoded) - compressed multipart uuencoded tar file
75. [NLM HyperDoc Visible Human Project](http://www.nlm.nih.gov/research/visible/visible_human.html) - Color, CAT and MRI image samples - over 30 images (Formats: jpeg)
76. [National Design Repository](http://www.designrepository.org) - Over 55,000 3D CAD and solid models of (mostly) mechanical/machined engineering designs. (Formats: gif,vrml,wrl,stp,sat) 
77. [Geometric & Intelligent Computing Laboratory](http://gicl.mcs.drexel.edu)
79. [OSU (MSU) 3D Object Model Database](http://eewww.eng.ohio-state.edu/~flynn/3DDB/Models/) - several sets of 3D object models collected over several years to use in object recognition research (Formats: homebrew, vrml)
80. [OSU (MSU/WSU) Range Image Database](http://eewww.eng.ohio-state.edu/~flynn/3DDB/RID/) - Hundreds of real and synthetic images (Formats: gif, homebrew)
81. [OSU/SAMPL Database: Range Images, 3D Models, Stills, Motion Sequences](http://sampl.eng.ohio-state.edu/~sampl/database.htm) - Over 1000 range images, 3D object models, still images and motion sequences (Formats: gif, ppm, vrml, homebrew)
82. [Signal Analysis and Machine Perception Laboratory](http://sampl.eng.ohio-state.edu)
84. [Otago Optical Flow Evaluation Sequences](http://www.cs.otago.ac.nz/research/vision/Research/OpticalFlow/opticalflow.html) - Synthetic and real sequences with machine-readable ground truth optical flow fields, plus tools to generate ground truth for new sequences. (Formats: ppm,tif,homebrew)
85. [Vision Research Group](http://www.cs.otago.ac.nz/research/vision/index.html)
87. [ftp://ftp.limsi.fr/pub/quenot/opflow/testdata/piv/](ftp://ftp.limsi.fr/pub/quenot/opflow/testdata/piv/) - Real and synthetic image sequences used for testing a Particle Image Velocimetry application. These images may be used for the test of optical flow and image matching algorithms. (Formats: pgm (raw))
88. [LIMSI-CNRS/CHM/IMM/vision](http://www.limsi.fr/Recherche/IMM/PageIMM.html)
89. [LIMSI-CNRS](http://www.limsi.fr/)
90. [Photometric 3D Surface Texture Database](http://www.taurusstudio.net/research/pmtexdb/index.htm) - This is the first 3D texture database which provides both full real surface rotations and registered photometric stereo data (30 textures, 1680 images). (Formats: TIFF)
91. [SEQUENCES FOR OPTICAL FLOW ANALYSIS (SOFA)](http://www.cee.hw.ac.uk/~mtc/sofa) - 9 synthetic sequences designed for testing motion analysis applications, including full ground truth of motion and camera parameters. (Formats: gif)
92. [Computer Vision Group](http://www.cee.hw.ac.uk/~mtc/research.html)
94. [Sequences for Flow Based Reconstruction](http://www.nada.kth.se/~zucch/CAMERA/PUB/seq.html) - synthetic sequence for testing structure from motion algorithms (Formats: pgm)
95. [Stereo Images with Ground Truth Disparity and Occlusion](http://www-dbv.cs.uni-bonn.de/stereo_data/) - a small set of synthetic images of a hallway with varying amounts of noise added. Use these images to benchmark your stereo algorithm. (Formats: raw, viff (khoros), or tiff)
96. [Stuttgart Range Image Database](http://range.informatik.uni-stuttgart.de) - A collection of synthetic range images taken from high-resolution polygonal models available on the web (Formats: homebrew)
97. [Department Image Understanding](http://www.informatik.uni-stuttgart.de/ipvr/bv/bv_home_engl.html)
99. [The AR Face Database](http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html) - Contains over 4,000 color images corresponding to 126 people's faces (70 men and 56 women). Frontal views with variations in facial expressions, illumination, and occlusions. (Formats: RAW (RGB 24-bit))
100. [Purdue Robot Vision Lab](http://rvl.www.ecn.purdue.edu/RVL/)
101. [The MIT-CSAIL Database of Objects and Scenes](http://web.mit.edu/torralba/www/database.html) - Database for testing multiclass object detection and scene recognition algorithms. Over 72,000 images with 2873 annotated frames. More than 50 annotated object classes. (Formats: jpg)
102. [The RVL SPEC-DB (SPECularity DataBase)](http://rvl1.ecn.purdue.edu/RVL/specularity_database/) - A collection of over 300 real images of 100 objects taken under three different illuminaiton conditions (Diffuse/Ambient/Directed). -- Use these images to test algorithms for detecting and compensating specular highlights in color images. (Formats: TIFF )
103. [Robot Vision Laboratory](http://rvl1.ecn.purdue.edu/RVL/)
105. [The Xm2vts database](http://xm2vtsdb.ee.surrey.ac.uk) - The XM2VTSDB contains four digital recordings of 295 people taken over a period of four months. This database contains both image and video data of faces.
106. [Centre for Vision, Speech and Signal Processing](http://www.ee.surrey.ac.uk/Research/CVSSP)
107. [Traffic Image Sequences and 'Marbled Block' Sequence](http://i21www.ira.uka.de/image_sequences) - thousands of frames of digitized traffic image sequences as well as the 'Marbled Block' sequence (grayscale images) (Formats: GIF)
108. [IAKS/KOGS](http://i21www.ira.uka.de)
110. [U Bern Face images](ftp://ftp.iam.unibe.ch/pub/Images/FaceImages) - hundreds of images (Formats: Sun rasterfile)
111. [U Michigan textures](ftp://freebie.engin.umich.edu/pub/misc/textures) (Formats: compressed raw)
112. [U Oulu wood and knots database](http://www.ee.oulu.fi/~olli/Projects/Lumber.Grading.html) - Includes classifications - 1000+ color images (Formats: ppm)
113. [UCID - an Uncompressed Colour Image Database](http://vision.doc.ntu.ac.uk/datasets/UCID/ucid.html) - a benchmark database for image retrieval with predefined ground truth. (Formats: tiff)
115. [UMass Vision Image Archive](http://vis-www.cs.umass.edu/~vislib/) - Large image database with aerial, space, stereo, medical images and more. (Formats: homebrew)
116. [UNC's 3D image database](ftp://sunsite.unc.edu/pub/academic/computer-science/virtual-reality/3d) - many images 

---

# HUGGINGFACE TOP MODELS

**Source:** https://huggingface.co/models
**Total:** 100 top models by downloads

| Rank | Model | Downloads | Likes | Tags |
|------|-------|-----------|-------|------|
| 1 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 135,521,460 | 4044 | sentence-transformers, pytorch, tf |
| 2 | [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) | 100,233,683 | 863 | transformers, pytorch, safetensors |
| 3 | [google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator) | 76,384,423 | 67 | transformers, pytorch, tf |
| 4 | [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | 54,812,136 | 2454 | transformers, pytorch, tf |
| 5 | [dima806/fairface_age_image_detection](https://huggingface.co/dima806/fairface_age_image_detection) | 53,418,523 | 47 | transformers, safetensors, vit |
| 6 | [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) | 21,135,747 | 253 | transformers, pytorch, tf |
| 7 | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | 20,846,524 | 794 | transformers, pytorch, tf |
| 8 | [timm/mobilenetv3_small_100.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k) | 19,002,995 | 40 | timm, pytorch, safetensors |
| 9 | [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) | 18,524,930 | 40 | transformers, pytorch, safetensors |
| 10 | [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 17,625,723 | 1180 | sentence-transformers, pytorch, onnx |
| 11 | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | 15,739,353 | 637 | pyannote-audio, pytorch, pyannote |
| 12 | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) | 15,710,918 | 530 | transformers, pytorch, tf |
| 13 | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) | 14,574,796 | 621 | ultralytics, pytorch, dataset:wider_face |
| 14 | [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) | 14,195,501 | 780 | transformers, pytorch, tf |
| 15 | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 14,015,751 | 1047 | sentence-transformers, pytorch, tf |
| 16 | [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | 13,930,406 | 81 | pyannote-audio, pytorch, pyannote |
| 17 | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | 13,585,862 | 56 | transformers, pytorch, tf |
| 18 | [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) | 13,560,481 | 131 | transformers, pytorch, BERT |
| 19 | [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | 13,138,955 | 1256 | pyannote-audio, pyannote, pyannote-audio-pipeline |
| 20 | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | 11,286,198 | 3003 | transformers, pytorch, tf |
| 21 | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | 10,646,146 | 1887 | transformers, pytorch, tf |
| 22 | [omni-research/Tarsier2-Recap-7b](https://huggingface.co/omni-research/Tarsier2-Recap-7b) | 9,875,588 | 19 | safetensors, video LLM, arxiv:2501.07888 |
| 23 | [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16) | 9,825,058 | 187 | moshi, safetensors, en |
| 24 | [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | 8,726,357 | 748 | transformers, pytorch, tf |
| 25 | [facebook/contriever](https://huggingface.co/facebook/contriever) | 8,166,904 | 69 | transformers, pytorch, bert |
| 26 | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 8,037,867 | 844 | transformers, safetensors, qwen2 |
| 27 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 7,285,383 | 745 | transformers, safetensors, qwen3 |
| 28 | [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) | 7,057,307 | 545 | transformers, pytorch, tf |
| 29 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 6,924,828 | 2450 | sentence-transformers, pytorch, onnx |
| 30 | [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | 6,470,494 | 294 | transformers, pytorch, onnx |
| 31 | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 6,468,099 | 544 | transformers, safetensors, qwen2_5_vl |
| 32 | [Gensyn/Qwen2.5-0.5B-Instruct](https://huggingface.co/Gensyn/Qwen2.5-0.5B-Instruct) | 6,438,470 | 26 | transformers, safetensors, qwen2 |
| 33 | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) | 6,297,817 | 417 | diffusion-single-file, comfyui, region:us |
| 34 | [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) | 6,269,193 | 154 | transformers, pytorch, safetensors |
| 35 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 6,068,447 | 378 | sentence-transformers, pytorch, onnx |
| 36 | [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) | 5,916,766 | 1088 | transformers, pytorch, onnx |
| 37 | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) | 5,914,521 | 1653 | diffusion-single-file, comfyui, base_model:Wan-AI/Wan2.1-VACE-1.3B |
| 38 | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | 5,847,261 | 381 | transformers, pytorch, tf |
| 39 | [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | 5,767,219 | 839 | transformers, pytorch, tf |
| 40 | [Isotonic/distilbert_finetuned_ai4privacy_v2](https://huggingface.co/Isotonic/distilbert_finetuned_ai4privacy_v2) | 5,742,416 | 21 | transformers, onnx, safetensors |
| 41 | [nlpaueb/legal-bert-base-uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased) | 5,582,479 | 283 | transformers, pytorch, tf |
| 42 | [jonatasgrosman/wav2vec2-large-xlsr-53-russian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian) | 5,547,834 | 62 | transformers, pytorch, jax |
| 43 | [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | 5,440,981 | 3131 | coqui, text-to-speech, license:other |
| 44 | [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) | 5,335,720 | 420 | sentence-transformers, pytorch, tf |
| 45 | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 5,260,341 | 4844 | transformers, safetensors, llama |
| 46 | [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 5,224,817 | 158 | sentence-transformers, pytorch, jax |
| 47 | [autogluon/chronos-bolt-small](https://huggingface.co/autogluon/chronos-bolt-small) | 5,199,150 | 18 | safetensors, t5, time series |
| 48 | [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | 5,182,772 | 274 | sentence-transformers, pytorch, rust |
| 49 | [dandelin/vilt-b32-mlm](https://huggingface.co/dandelin/vilt-b32-mlm) | 5,008,446 | 11 | transformers, pytorch, vilt |
| 50 | [google-bert/bert-base-multilingual-uncased](https://huggingface.co/google-bert/bert-base-multilingual-uncased) | 4,967,896 | 140 | transformers, pytorch, tf |
| 51 | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 4,751,782 | 3824 | transformers, safetensors, gpt_oss |
| 52 | [dphn/dolphin-2.9.1-yi-1.5-34b](https://huggingface.co/dphn/dolphin-2.9.1-yi-1.5-34b) | 4,724,964 | 44 | transformers, safetensors, llama |
| 53 | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) | 4,671,479 | 774 | diffusion-single-file, comfyui, region:us |
| 54 | [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) | 4,631,895 | 145 | sentence-transformers, pytorch, tf |
| 55 | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | 4,599,606 | 593 | sentence-transformers, pytorch, onnx |
| 56 | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 4,532,452 | 677 | transformers, safetensors, gemma3_text |
| 57 | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 4,508,805 | 365 | sentence-transformers, pytorch, onnx |
| 58 | [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | 4,465,516 | 727 | transformers, pytorch, tf |
| 59 | [tech4humans/yolov8s-signature-detector](https://huggingface.co/tech4humans/yolov8s-signature-detector) | 4,375,124 | 48 | ultralytics, tensorboard, onnx |
| 60 | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 4,342,260 | 694 | sentence-transformers, safetensors, qwen3 |
| 61 | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | 4,335,764 | 5206 | text-to-speech, en, arxiv:2306.07691 |
| 62 | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | 4,329,979 | 1437 | transformers, safetensors, llama |
| 63 | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 4,270,087 | 5033 | transformers, pytorch, jax |
| 64 | [neulab/codebert-python](https://huggingface.co/neulab/codebert-python) | 4,229,293 | 26 | transformers, pytorch, safetensors |
| 65 | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 4,222,552 | 2657 | transformers, safetensors, whisper |
| 66 | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 4,217,697 | 533 | transformers, safetensors, qwen2 |
| 67 | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 4,174,196 | 1326 | transformers, safetensors, qwen2_5_vl |
| 68 | [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) | 4,148,297 | 222 | transformers, pytorch, tf |
| 69 | [trl-internal-testing/tiny-Qwen2ForCausalLM-2.5](https://huggingface.co/trl-internal-testing/tiny-Qwen2ForCausalLM-2.5) | 3,973,781 | 1 | transformers, safetensors, qwen2 |
| 70 | [MahmoudAshraf/mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) | 3,951,817 | 62 | transformers, pytorch, safetensors |
| 71 | [autogluon/chronos-bolt-base](https://huggingface.co/autogluon/chronos-bolt-base) | 3,932,010 | 30 | safetensors, t5, time series |
| 72 | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 3,918,395 | 1137 | transformers, safetensors, llama |
| 73 | [hustvl/vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k) | 3,861,447 | 46 | transformers, pytorch, safetensors |
| 74 | [jonatasgrosman/wav2vec2-large-xlsr-53-portuguese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-portuguese) | 3,851,830 | 35 | transformers, pytorch, jax |
| 75 | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3,821,561 | 322 | transformers, safetensors, qwen2 |
| 76 | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | 3,747,665 | 4077 | transformers, safetensors, gpt_oss |
| 77 | [sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1) | 3,711,731 | 183 | sentence-transformers, pytorch, onnx |
| 78 | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 3,703,041 | 430 | transformers, safetensors, qwen3 |
| 79 | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) | 3,676,037 | 423 | diffusers, safetensors, text-to-image |
| 80 | [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k) | 3,668,584 | 39 | timm, pytorch, safetensors |
| 81 | [jonatasgrosman/wav2vec2-large-xlsr-53-japanese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese) | 3,609,238 | 42 | transformers, pytorch, jax |
| 82 | [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) | 3,596,585 | 392 | transformers, pytorch, tf |
| 83 | [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) | 3,596,016 | 277 | transformers, pytorch, tf |
| 84 | [datasocietyco/bge-base-en-v1.5-course-recommender-v5](https://huggingface.co/datasocietyco/bge-base-en-v1.5-course-recommender-v5) | 3,579,656 | 1 | sentence-transformers, safetensors, bert |
| 85 | [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) | 3,560,724 | 228 | transformers, onnx, safetensors |
| 86 | [facebook/esmfold_v1](https://huggingface.co/facebook/esmfold_v1) | 3,558,017 | 43 | transformers, pytorch, esm |
| 87 | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 3,465,975 | 779 | sentence-transformers, safetensors, xlm-roberta |
| 88 | [google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased) | 3,400,185 | 334 | transformers, pytorch, tf |
| 89 | [timm/convnextv2_nano.fcmae_ft_in22k_in1k](https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k) | 3,384,615 | 2 | timm, pytorch, safetensors |
| 90 | [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) | 3,304,724 | 1477 | transformers, pytorch, jax |
| 91 | [patrickjohncyh/fashion-clip](https://huggingface.co/patrickjohncyh/fashion-clip) | 3,275,210 | 247 | transformers, pytorch, onnx |
| 92 | [w11wo/indonesian-roberta-base-posp-tagger](https://huggingface.co/w11wo/indonesian-roberta-base-posp-tagger) | 3,255,485 | 9 | transformers, pytorch, tf |
| 93 | [bigscience/bloomz-560m](https://huggingface.co/bigscience/bloomz-560m) | 3,191,538 | 129 | transformers, pytorch, tensorboard |
| 94 | [context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16](https://huggingface.co/context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16) | 3,141,547 | 7 | transformers, safetensors, llama |
| 95 | [ggml-org/models-moved](https://huggingface.co/ggml-org/models-moved) | 3,118,259 | 12 | gguf, endpoints_compatible, region:us |
| 96 | [OpenGVLab/InternVL3_5-241B-A28B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B-Instruct) | 3,101,117 | 15 | transformers, safetensors, internvl_chat |
| 97 | [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) | 3,046,638 | 1486 | transformers, pytorch, tf |
| 98 | [google-t5/t5-small](https://huggingface.co/google-t5/t5-small) | 3,027,461 | 500 | transformers, pytorch, tf |
| 99 | [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) | 2,977,448 | 142 | transformers, pytorch, jax |
| 100 | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) | 2,892,809 | 885 | transformers, pytorch, tf |

---

# HUGGINGFACE TOP DATASETS

**Source:** https://huggingface.co/datasets
**Total:** 100 top datasets by downloads

| Rank | Dataset | Downloads | Likes | Tags |
|------|---------|-----------|-------|------|
| 1 | [nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) | 2,772,783 | 36 | task_categories:other, license:cc-by-4.0, size_categories:10K<n<100K |
| 2 | [AquaV/genshin-voices-separated](https://huggingface.co/datasets/AquaV/genshin-voices-separated) | 2,378,414 | 8 | region:us |
| 3 | [banned-historical-archives/banned-historical-archives](https://huggingface.co/datasets/banned-historical-archives/banned-historical-archives) | 1,984,171 | 5 | size_categories:n<1K, format:imagefolder, modality:image |
| 4 | [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images) | 1,964,278 | 89 | license:cc-by-nc-sa-4.0, size_categories:n<1K, format:imagefolder |
| 5 | [nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | 1,842,851 | 159 | task_categories:robotics, license:cc-by-4.0, region:us |
| 6 | [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | 1,465,361 | 20 | license:mit, size_categories:1K<n<10K, format:parquet |
| 7 | [tasl-lab/uniocc](https://huggingface.co/datasets/tasl-lab/uniocc) | 1,100,022 | 2 | task_categories:image-to-3d, license:mit, modality:3d |
| 8 | [princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | 990,714 | 223 | size_categories:n<1K, format:parquet, modality:text |
| 9 | [Salesforce/GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain) | 940,599 | 15 | task_categories:time-series-forecasting, license:apache-2.0, size_categories:1M<n<10M |
| 10 | [lavita/medical-qa-shared-task-v1-toy](https://huggingface.co/datasets/lavita/medical-qa-shared-task-v1-toy) | 901,048 | 21 | size_categories:n<1K, format:parquet, modality:tabular |
| 11 | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) | 893,833 | 513 | task_categories:text-generation, task_categories:fill-mask, task_ids:language-modeling |
| 12 | [chcorbi/helvipad](https://huggingface.co/datasets/chcorbi/helvipad) | 781,772 | 10 | task_categories:depth-estimation, source_datasets:original, license:cc0-1.0 |
| 13 | [hf-doc-build/doc-build](https://huggingface.co/datasets/hf-doc-build/doc-build) | 728,633 | 12 | license:mit, region:us |
| 14 | [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | 632,255 | 240 | license:cc-by-4.0, arxiv:2406.11794, region:us |
| 15 | [allenai/c4](https://huggingface.co/datasets/allenai/c4) | 624,932 | 477 | task_categories:text-generation, task_categories:fill-mask, task_ids:language-modeling |
| 16 | [IPEC-COMMUNITY/language_table_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/language_table_lerobot) | 590,894 | 0 | task_categories:robotics, license:apache-2.0, region:us |
| 17 | [allenai/objaverse](https://huggingface.co/datasets/allenai/objaverse) | 537,327 | 408 | language:en, license:odc-by, arxiv:2212.08051 |
| 18 | [applied-ai-018/pretraining_v1-omega_books](https://huggingface.co/datasets/applied-ai-018/pretraining_v1-omega_books) | 514,605 | 2 | size_categories:100M<n<1B, format:parquet, modality:tabular |
| 19 | [wyu1/Leopard-Instruct](https://huggingface.co/datasets/wyu1/Leopard-Instruct) | 448,603 | 63 | language:en, license:apache-2.0, size_categories:1M<n<10M |
| 20 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 432,493 | 923 | annotations_creators:crowdsourced, language_creators:crowdsourced, multilinguality:monolingual |
| 21 | [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) | 386,824 | 448 | task_categories:text-classification, task_ids:acceptability-classification, task_ids:natural-language-inference |
| 22 | [Helsinki-NLP/fineweb-edu-translated](https://huggingface.co/datasets/Helsinki-NLP/fineweb-edu-translated) | 348,691 | 1 | task_categories:text-generation, language:bg, language:ca |
| 23 | [Symato/cc](https://huggingface.co/datasets/Symato/cc) | 334,948 | 2 | language:vi, license:mit, size_categories:1K<n<10K |
| 24 | [huggingface/badges](https://huggingface.co/datasets/huggingface/badges) | 331,113 | 46 | license:mit, size_categories:n<1K, format:imagefolder |
| 25 | [permutans/fineweb-bbc-news](https://huggingface.co/datasets/permutans/fineweb-bbc-news) | 314,523 | 23 | language:en, license:odc-by, size_categories:10M<n<100M |
| 26 | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 314,354 | 789 | task_categories:text-generation, language:en, license:odc-by |
| 27 | [IPEC-COMMUNITY/kuka_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/kuka_lerobot) | 313,835 | 0 | task_categories:robotics, license:apache-2.0, modality:video |
| 28 | [huggingface-course/documentation-images](https://huggingface.co/datasets/huggingface-course/documentation-images) | 307,997 | 0 | license:apache-2.0, size_categories:n<1K, format:imagefolder |
| 29 | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 293,189 | 2410 | task_categories:text-generation, language:en, license:odc-by |
| 30 | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | 269,994 | 570 | task_categories:question-answering, task_ids:multiple-choice-qa, annotations_creators:no-annotation |
| 31 | [espnet/yodas2](https://huggingface.co/datasets/espnet/yodas2) | 254,204 | 41 | license:cc-by-3.0, arxiv:2406.00899, region:us |
| 32 | [pmchard/3D-ADAM](https://huggingface.co/datasets/pmchard/3D-ADAM) | 251,512 | 3 | language:en, license:cc-by-nc-sa-4.0, size_categories:1K<n<10K |
| 33 | [HuggingFaceM4/FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) | 242,559 | 422 | size_categories:10M<n<100M, format:parquet, modality:image |
| 34 | [mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M) | 240,287 | 45 | license:apache-2.0, size_categories:10M<n<100M, format:parquet |
| 35 | [openclimatefix/met-office-uk-deterministic-solar](https://huggingface.co/datasets/openclimatefix/met-office-uk-deterministic-solar) | 236,495 | 1 | task_categories:time-series-forecasting, task_ids:multivariate-time-series-forecasting, annotations_creators:expert-generated |
| 36 | [mvp-lab/LLaVA-OneVision-1.5-Instruct-Data](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data) | 236,285 | 53 | task_categories:image-text-to-text, language:en, license:apache-2.0 |
| 37 | [behavior-1k/2025-challenge-demos](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos) | 232,536 | 18 | task_categories:robotics, license:mit, size_categories:100M<n<1B |
| 38 | [Gourieff/ReActor](https://huggingface.co/datasets/Gourieff/ReActor) | 218,207 | 149 | license:mit, region:us |
| 39 | [agents-course/course-images](https://huggingface.co/datasets/agents-course/course-images) | 203,605 | 16 | size_categories:n<1K, format:imagefolder, modality:image |
| 40 | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | 202,291 | 222 | task_categories:question-answering, task_ids:open-domain-qa, task_ids:multiple-choice-qa |
| 41 | [hf-doc-build/doc-build-dev](https://huggingface.co/datasets/hf-doc-build/doc-build-dev) | 201,820 | 6 | license:mit, region:us, documentation |
| 42 | [jat-project/jat-dataset](https://huggingface.co/datasets/jat-project/jat-dataset) | 201,232 | 44 | task_categories:reinforcement-learning, task_categories:text-generation, task_categories:question-answering |
| 43 | [zcbecda/SpineAlign](https://huggingface.co/datasets/zcbecda/SpineAlign) | 199,275 | 0 | size_categories:1K<n<10K, format:imagefolder, modality:3d |
| 44 | [Felix92/docTR-resource-collection](https://huggingface.co/datasets/Felix92/docTR-resource-collection) | 192,721 | 1 | license:apache-2.0, size_categories:1K<n<10K, format:text |
| 45 | [mteb/sts22-crosslingual-sts](https://huggingface.co/datasets/mteb/sts22-crosslingual-sts) | 191,702 | 7 | task_categories:sentence-similarity, task_ids:semantic-similarity-scoring, annotations_creators:human-annotated |
| 46 | [ming030890/youtube_caption_yue](https://huggingface.co/datasets/ming030890/youtube_caption_yue) | 188,557 | 0 | size_categories:10K<n<100K, format:parquet, modality:audio |
| 47 | [adams-story/datacomp200m](https://huggingface.co/datasets/adams-story/datacomp200m) | 185,828 | 2 | size_categories:100M<n<1B, format:parquet, modality:image |
| 48 | [HPLT/HPLT2.0_cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | 185,691 | 36 | task_categories:fill-mask, task_categories:text-generation, task_ids:language-modeling |
| 49 | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) | 180,441 | 347 | task_categories:text-classification, task_ids:sentiment-classification, annotations_creators:expert-generated |
| 50 | [xlangai/ubuntu_osworld_file_cache](https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache) | 166,259 | 2 | license:apache-2.0, arxiv:2404.07972, region:us |
| 51 | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | 165,091 | 149 | language:en, size_categories:10K<n<100K, format:parquet |
| 52 | [Zyphra/Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2) | 158,672 | 84 | task_categories:text-generation, language:en, license:odc-by |
| 53 | [open-ko-llm-leaderboard/requests-backup](https://huggingface.co/datasets/open-ko-llm-leaderboard/requests-backup) | 154,625 | 5 | region:us |
| 54 | [jat-project/jat-dataset-tokenized](https://huggingface.co/datasets/jat-project/jat-dataset-tokenized) | 142,899 | 1 | size_categories:10M<n<100M, format:parquet, modality:timeseries |
| 55 | [IPEC-COMMUNITY/bridge_orig_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot) | 142,014 | 7 | task_categories:robotics, license:apache-2.0, modality:video |
| 56 | [m-a-p/FineFineWeb](https://huggingface.co/datasets/m-a-p/FineFineWeb) | 139,348 | 83 | task_categories:text-classification, task_categories:text-generation, language:en |
| 57 | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | 136,036 | 174 | task_categories:text-classification, task_ids:topic-classification, annotations_creators:found |
| 58 | [mbouhaja/AI_REAL](https://huggingface.co/datasets/mbouhaja/AI_REAL) | 135,635 | 0 | task_categories:image-classification, annotations_creators:manual, language:en |
| 59 | [aps/super_glue](https://huggingface.co/datasets/aps/super_glue) | 133,303 | 176 | task_categories:text-classification, task_categories:token-classification, task_categories:question-answering |
| 60 | [Stevross/mmlu](https://huggingface.co/datasets/Stevross/mmlu) | 126,958 | 8 | task_categories:question-answering, task_ids:multiple-choice-qa, annotations_creators:no-annotation |
| 61 | [espnet/yodas-granary](https://huggingface.co/datasets/espnet/yodas-granary) | 124,705 | 21 | task_categories:automatic-speech-recognition, task_categories:translation, language:bg |
| 62 | [NickL77/Llama3.1-8B-BaldEagle3-Ultrachat](https://huggingface.co/datasets/NickL77/Llama3.1-8B-BaldEagle3-Ultrachat) | 124,022 | 1 | region:us |
| 63 | [IPEC-COMMUNITY/droid_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot) | 122,356 | 6 | task_categories:robotics, license:apache-2.0, region:us |
| 64 | [mcp-course/images](https://huggingface.co/datasets/mcp-course/images) | 122,034 | 5 | license:apache-2.0, size_categories:n<1K, format:imagefolder |
| 65 | [cadene/droid_1.0.1](https://huggingface.co/datasets/cadene/droid_1.0.1) | 119,543 | 8 | task_categories:robotics, license:apache-2.0, region:us |
| 66 | [TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-2024-08-06](https://huggingface.co/datasets/TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-2024-08-06) | 117,893 | 1 | size_categories:10K<n<100K, format:parquet, modality:text |
| 67 | [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) | 117,050 | 72 | language:en, size_categories:10K<n<100K, format:parquet |
| 68 | [updatebao/country](https://huggingface.co/datasets/updatebao/country) | 116,260 | 0 | size_categories:1K<n<10K, format:imagefolder, modality:image |
| 69 | [hf-internal-testing/transformers_circleci_workflow_runs](https://huggingface.co/datasets/hf-internal-testing/transformers_circleci_workflow_runs) | 116,091 | 0 | region:us |
| 70 | [shi-labs/oneformer_demo](https://huggingface.co/datasets/shi-labs/oneformer_demo) | 113,901 | 0 | region:us |
| 71 | [huggingface/brand-assets](https://huggingface.co/datasets/huggingface/brand-assets) | 112,867 | 7 | size_categories:n<1K, format:imagefolder, modality:image |
| 72 | [hallucinations-leaderboard/results](https://huggingface.co/datasets/hallucinations-leaderboard/results) | 112,791 | 2 | license:apache-2.0, region:us |
| 73 | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 107,095 | 676 | task_categories:text-generation, language:aai, language:aak |
| 74 | [japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized](https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized) | 106,489 | 0 | size_categories:1M<n<10M, format:parquet, library:datasets |
| 75 | [nunchaku-tech/cdn](https://huggingface.co/datasets/nunchaku-tech/cdn) | 106,084 | 0 | license:apache-2.0, size_categories:n<1K, format:imagefolder |
| 76 | [jamesqijingsong/zidian](https://huggingface.co/datasets/jamesqijingsong/zidian) | 105,572 | 0 | language:zh, language:en, license:cc-by-nc-4.0 |
| 77 | [dlxjj/pdf_ocr](https://huggingface.co/datasets/dlxjj/pdf_ocr) | 103,905 | 0 | size_categories:n<1K, format:imagefolder, modality:image |
| 78 | [huggingface-deep-rl-course/course-images](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images) | 99,728 | 0 | region:us |
| 79 | [daniilak/nbchr_pdfs](https://huggingface.co/datasets/daniilak/nbchr_pdfs) | 99,124 | 0 | license:unknown, size_categories:10K<n<100K, library:datasets |
| 80 | [hf-vision/course-assets](https://huggingface.co/datasets/hf-vision/course-assets) | 97,693 | 9 | license:apache-2.0, size_categories:n<1K, format:imagefolder |
| 81 | [amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) | 97,393 | 392 | task_categories:text-to-speech, task_categories:automatic-speech-recognition, language:zh |
| 82 | [KakologArchives/KakologArchives](https://huggingface.co/datasets/KakologArchives/KakologArchives) | 96,366 | 16 | task_categories:text-classification, language:ja, license:mit |
| 83 | [open-llm-leaderboard-old/requests](https://huggingface.co/datasets/open-llm-leaderboard-old/requests) | 93,580 | 22 | license:apache-2.0, size_categories:n<1K, format:json |
| 84 | [CQSB/SoftDis](https://huggingface.co/datasets/CQSB/SoftDis) | 93,005 | 0 | size_categories:100K<n<1M, modality:text, modality:timeseries |
| 85 | [updatebao/geonamebase_1](https://huggingface.co/datasets/updatebao/geonamebase_1) | 90,206 | 0 | modality:image, region:us |
| 86 | [ComputerVisionAnimeProject/AnimeFaceColorization](https://huggingface.co/datasets/ComputerVisionAnimeProject/AnimeFaceColorization) | 89,228 | 0 | language:en, license:cc-by-4.0, size_categories:10K<n<100K |
| 87 | [InternRobotics/InternData-fractal20220817_data](https://huggingface.co/datasets/InternRobotics/InternData-fractal20220817_data) | 88,182 | 0 | size_categories:1K<n<10K, format:json, modality:tabular |
| 88 | [mlfoundations/MINT-1T-PDF-CC-2024-18](https://huggingface.co/datasets/mlfoundations/MINT-1T-PDF-CC-2024-18) | 88,078 | 19 | task_categories:image-to-text, task_categories:text-generation, language:en |
| 89 | [locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU) | 87,073 | 42 | task_categories:question-answering, task_ids:closed-domain-qa, annotations_creators:machine-generated |
| 90 | [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) | 85,865 | 340 | task_categories:question-answering, task_ids:extractive-qa, annotations_creators:crowdsourced |
| 91 | [huggingchat/models-logo](https://huggingface.co/datasets/huggingchat/models-logo) | 85,860 | 5 | size_categories:n<1K, format:imagefolder, modality:image |
| 92 | [ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | 84,472 | 177 | task_categories:image-to-text, task_categories:text-to-image, task_categories:image-classification |
| 93 | [open-llm-leaderboard/requests](https://huggingface.co/datasets/open-llm-leaderboard/requests) | 84,136 | 12 | license:apache-2.0, region:us |
| 94 | [SVCFusion/Launcher](https://huggingface.co/datasets/SVCFusion/Launcher) | 83,649 | 0 | license:cc, region:us |
| 95 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | 83,130 | 344 | annotations_creators:expert-generated, language_creators:expert-generated, multilinguality:monolingual |
| 96 | [zhoujt1994/HumanCellEpigenomeAtlas_sc_contact](https://huggingface.co/datasets/zhoujt1994/HumanCellEpigenomeAtlas_sc_contact) | 83,113 | 0 | license:mit, region:us |
| 97 | [hf-internal-testing/hf_hub_cache](https://huggingface.co/datasets/hf-internal-testing/hf_hub_cache) | 82,564 | 0 | region:us |
| 98 | [maknee/leaague-of-legends-decoded-replay-packets-s12-unorganized](https://huggingface.co/datasets/maknee/leaague-of-legends-decoded-replay-packets-s12-unorganized) | 82,206 | 0 | license:apache-2.0, region:us |
| 99 | [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) | 81,663 | 300 | task_categories:summarization, task_ids:news-articles-summarization, annotations_creators:no-annotation |
| 100 | [Narsil/image_dummy](https://huggingface.co/datasets/Narsil/image_dummy) | 81,414 | 0 | size_categories:n<1K, modality:audio, modality:image |

---

## 📊 COLLECTION SUMMARY

### Total Statistics

- **Total Resources:** 2,229+
- **Total Platforms:** 5
- **Total Stars (source repos):** 102,000+

### By Platform

| Platform | Resources | Stars |
|----------|-----------|-------|
| Awesome Machine Learning | 1,268 | 65,000 |
| Awesome Deep Learning | 610 | 23,000 |
| Transferlearning | 151 | 14,000 |
| HuggingFace Models | 100 | - |
| HuggingFace Datasets | 100 | - |
| **TOTAL** | **2,229** | **102,000+** |

---

**Created:** October 30, 2025
**Source:** Consolidated from 5 major AI platforms
