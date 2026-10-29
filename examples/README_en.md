English | [简体中文](./README.md)

## Overview

> Note:
> - Currently, most operators in the operator library run on AI Core, and a few run on AI CPU. By default, an operator mentioned in this project refers to an AI Core operator.
> - For the introduction to AI Core and AI CPU, see the "Concepts, Principles, and Terms > Hardware Architecture and Data Processing Principles" section in [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC).

This project provides development and invocation samples for AI Core operators and AI CPU operators. Refer to the corresponding implementations based on actual requirements.

## Directory Description
```
├── examples
│   ├── add_example                # AI Core operator name
│   │   ├── CMakeLists.txt         # Operator compilation configuration file; keep the original file
│   │   ├── examples               # Operator usage samples
│   │   ├── op_graph               # Directory related to operator graph construction
│   │   ├── op_host                # Implementation of the operator information library, Tiling, and InferShape
│   │   └── op_kernel              # Operator kernel directory
│   ├── add_example_aicpu          # AI CPU operator name
│   │   ├── CMakeLists.txt         # Operator compilation configuration file; keep the original file
│   │   ├── examples               # Operator usage samples
│   │   ├── op_graph               # Directory related to operator graph construction
│   │   ├── op_host                # Implementation of the operator information library and InferShape
│   │   └── op_kernel_aicpu        # Operator kernel directory
│   ├── CMakeLists.txt             # Operator compilation configuration file; keep the original file
│   └── README.md                  # Operator description document

```

## Operator Development Samples
|Sample Directory|Sample Description|Operator Development|Operator Invocation|
|---|------------------|---|---|
| add_example | Operator that implements the addition of two tensors. | For the end-to-end operator development process, see [AI Core Operator Development Guide](../docs/en/develop/aicore_develop_guide.md). | For invocation samples, see [examples](./add_example/examples/). |
|add_example_aicpu| Operator that implements the addition of two tensors. | For the end-to-end operator development process, see [AI CPU Operator Development Guide](../docs/en/develop/aicpu_develop_guide.md).| For invocation samples, see [examples](./add_example_aicpu/examples/). |
|fast_kernel_launch_example| Sample that implements fast end-to-end operator development in PyTorch scenarios. | For the end-to-end operator development process, see [PyTorch Operator Quick Development Guide](./fast_kernel_launch_example/README_en.md).| For invocation samples, see [examples](./fast_kernel_launch_example/). |
