# ML Framework Projects

This directory covers project examples organized around major ML frameworks and runtimes.

## Framework Families

- PyTorch, TensorFlow, JAX, scikit-learn, XGBoost, LightGBM, CatBoost, Spark ML, Ray, Hugging Face, ONNX Runtime, and domain-specific frameworks.
- Training frameworks such as Lightning, Accelerate, DeepSpeed, FSDP, Keras, and fastai.
- Serving and packaging frameworks such as BentoML, KServe, Seldon, Ray Serve, TorchServe, TensorFlow Serving, Triton, and vLLM.

## Reference Links

- PyTorch: https://pytorch.org/docs/stable/index.html
- TensorFlow: https://www.tensorflow.org/
- JAX: https://docs.jax.dev/
- scikit-learn: https://scikit-learn.org/stable/user_guide.html
- Hugging Face Accelerate: https://huggingface.co/docs/accelerate/index
- ONNX Runtime: https://onnxruntime.ai/docs/

## Project Metadata

Each project should record framework version, hardware, dataset, model architecture, training command, inference path, export format, metric, and reproducibility notes.

## Evaluation Standard

Include a baseline, a framework-specific advantage, known portability issues, and deployment compatibility. Do not treat a framework demo as production-ready unless it includes repeatable data, metrics, and validation.

## Routing Rules

- Put framework documentation links in the source atlas if they are general references.
- Put model-specific examples under the relevant model family.
- Put deployment-specific examples in `../MLOps_Projects/`.
