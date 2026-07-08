# Equivariant Neural Networks (SE(3) Equivariance & Steerable CNNs)

## Description

Equivariant Neural Networks (ENNs) are a class of deep learning architectures designed to incorporate geometric symmetries inherent to the data, such as rotation and translation. **SE(3) equivariance** refers to equivariance under the group of three-dimensional rigid motions (Special Euclidean Group in 3D), which includes rotations and translations. This ensures that the internal representation of an object changes in a predictable (equivariant) way when the object is transformed, and that the network's final output is invariant or equivariant to the input transformation. **Steerable CNNs** are a framework that implements this equivariance, where convolutional filters are constrained to be 'steerable', meaning that any rotated version of a filter can be expressed as a linear combination of a basis set of filters. This results in networks that are more data-efficient and have better generalization capability.

## Statistics

The main statistical benefit is **data efficiency** and **better generalization**. On benchmarks such as CIFAR, Steerable CNNs achieved state-of-the-art results with much more efficient parameter usage than traditional CNNs. In molecular modeling tasks (such as with NequIP), E(3) equivariance (which includes SE(3)) demonstrated significant performance gains, especially in limited-data scenarios. The computational cost may be higher compared to plain CNNs due to the complexity of equivariant convolution operations, but the gain in parameter efficiency and generalization generally compensates for it.

## Features

1. **Intrinsic Geometric Equivariance:** Ensures that the network respects the symmetries of the data (e.g., rotation, translation). 2. **Steerable Filters:** Allows convolutional filters to be transformed in a predictable way, reducing the number of independent parameters. 3. **Parameter Efficiency:** Uses parameters more efficiently, since symmetry knowledge is encoded in the architecture. 4. **Better Generalization:** The ability to generalize to unseen input transformations is significantly improved. 5. **3D Applications:** Essential for 3D data, such as point clouds and molecular structures, where orientation is arbitrary.

## Use Cases

1. **Computational Chemistry and Physics:** Modeling interatomic potentials (e.g., NequIP) and predicting molecular properties, where energy and other properties must be invariant to rotation and translation. 2. **3D Computer Vision:** Analysis of 3D point clouds, object recognition, and segmentation, where the object pose is arbitrary. 3. **Robotics and Manipulation:** SE(3)-Equivariant object representations for manipulation and motion planning tasks. 4. **Biomedical Image Analysis:** Segmentation of microscopy images and analysis of biological structures, where the sample orientation may vary. 5. **2D Image Classification:** Although more common in 3D, steerable CNNs (e.g., E(2)-equivariant) improve 2D classification by handling rotations.

## Integration

Integration is typically done through deep learning libraries that implement equivariant operations. \n\n**Library Example:** `escnn` (Equivariant Steerable CNNs) for PyTorch, the successor to `e2cnn`.\n\n**Installation (schematic):**\n```bash\npip install escnn\n```\n\n**Code Example (Equivariant Convolution in PyTorch with `escnn`):**\n```python\nimport torch\nfrom escnn import gspaces, nn\n\n# 1. Define the symmetry space (e.g., C4 for 90-degree rotations)\ngspace = gspaces.Rot2dOnR2(N=4)\n\n# 2. Define the input field type (e.g., a scalar field with 1 channel)\nin_type = nn.FieldType(gspace, [gspace.trivial_repr])\n\n# 3. Define the output field type (e.g., a regular field with 8 channels)\nout_type = nn.FieldType(gspace, [gspace.regular_repr] * 8)\n\n# 4. Create the equivariant convolutional layer\n# The layer ensures that the output transforms in a predictable way under the rotations of the C4 group\nconv = nn.R2Conv(in_type, out_type, kernel_size=5, padding=2)\n\n# 5. Create an input tensor (batch_size, channels, height, width)\nx = torch.randn(1, 1, 32, 32)\n\n# 6. Convert the tensor into a FieldType object for the network\nx_field = in_type(x)\n\n# 7. Apply the convolution\ny_field = conv(x_field)\n\n# 8. Get the output tensor\ny = y_field.tensor\n\nprint(f\"Input shape: {x.shape}\")\nprint(f\"Output shape: {y.shape}\")\n```

## URL

https://github.com/QUVA-Lab/escnn (escnn - Successor to e2cnn)
