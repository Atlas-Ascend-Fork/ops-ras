# AscendOps

English | [简体中文](./README.md)

## Prerequisites

- Before you start, complete the basic environment setup by referring to [Environment Setup](../../docs/en/install/quick_install.md).
- GCC 9.4.0+
- Python 3.8+
- PyTorch>=2.6.0
- [TorchNPU](https://gitcode.com/Ascend/pytorch/releases) of the matching version

## Installation Steps

1. Install Dependencies:
    ```sh
    python3 -m pip install -r requirements.txt
    ```

2. Build the Wheel:
    ```sh
    # -n: non-isolated build (uses existing environment)
    python3 -m build --wheel -n
    ```

3. Install Package:
    ```sh
    python3 -m pip install dist/*.whl --force-reinstall --no-deps
    ```

4. (Optional) Before rebuilding, it is recommended that you run the following command to clear the compilation cache:
   ```sh
   python3 setup.py clean
   ```

## Quick Start
After the installation, you can use NPU operators in the same way as ordinary PyTorch operations.

```python
import torch
import torch_npu
import ascend_ops

# Initialize data on NPU
x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()

# Call the custom NPU operator
npu_result = torch.ops.ascend_ops.add(x, y)

# Verify against CPU ATen implementation
cpu_x = x.cpu()
cpu_y = y.cpu()
cpu_result = cpu_x + cpu_y

assert torch.allclose(cpu_result, npu_result.cpu(), rtol=1e-6)
print("Verification successful!")
```

## Developer Guide: Adding a New Operator

To implement a new operator (for example, `add`), you only need to provide a C++ implementation.

1. First, create a folder named after the operator `add` in the csrc directory. In this folder, create a subfolder `ascend910b` named after the SoC for which you want to develop.

2. Create a `CMakeLists.txt` file in the soc directory:
    ```
    add_sources("--npu-arch=dav-2201")
    ```
    Here, `dav-2201` is the compilation parameter corresponding to the ascend910b chip.

3. Create an `add.cpp` file in the soc directory (using the operator name as the file name is recommended). This file contains all modules required for developing an AI Core operator.
    - Operator schema registration
    - Operator Meta Function implementation & registration
    - Operator kernel implementation (AscendC)
    - Operator NPU invocation implementation & registration

    ```cpp
    #include <ATen/Operators.h>
    #include <torch/all.h>
    #include <torch/library.h>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/framework/OpCommand.h"
    #include "kernel_operator.h"
    #include "platform/platform_ascendc.h"
    #include <type_traits>

    namespace ascend_ops {  // The current project is a namespace
    namespace Add {         // Each operator is recommended to have its own namespace to prevent global variable pollution

    /**
     * Register the operator schema with the PyTorch framework
     * so that the framework knows that this operator exists
     */
    // Register the operator's schema
    TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
    {
        m.def("add(Tensor x, Tensor y) -> Tensor");
    }

    /**
     * Implement the Meta function of the operator, that is, InferShape+InferDtype
     * Infer the output and the required space of the operator from the inputs, without actually computing the operator
     */
    // Meta function implementation of Add
    torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
    {
        TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
        auto z = torch::empty_like(x);
        return z;
    }

    /**
     * Register the Meta function of the operator with the framework
     * The framework can call this Meta function to know the required space before actually executing the operator computation
     * This lays the foundation for graph acceleration such as torch.compile/AutoGrad/AclGraph
     */
    // Register the Meta implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
    {
        m.impl("add", add_meta);
    }

    /**
     * NPU operator kernel implementation, written in AscendC APIs for the current soc
     */
    template <typename T>
    __global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        // kernel implementation
    }

    /**
     * Implement the operator invocation interface
     * In this interface, the NPU kernel invocation needs to be completed
     * 1. Compute the number/shape/dtype of the output tensors (by calling the Meta function or implementing directly)
     * 2. Compute Tiling: determine how to split the computation into blocks based on the shape
     * 3. Invoke the NPU kernel
     *
     */
    torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
    {
        auto z = add_meta(x, y);
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        int64_t totalLength, blockDim, blockLength, tileSize;
        totalLength = x.numel();
        std::tie(blockDim, blockLength, tileSize) = calc_tiling_params(totalLength);
        auto x_ptr = (GM_ADDR)x.data_ptr();
        auto y_ptr = (GM_ADDR)y.data_ptr();
        auto z_ptr = (GM_ADDR)z.data_ptr();
        auto acl_call = [=]() -> int {
            AT_DISPATCH_SWITCH(
                x.scalar_type(), "add_npu",
                // Invoke different NPU kernels for different data types
                AT_DISPATCH_CASE(torch::kFloat32, [&] {
                    using scalar_t = float;
                    add_kernel<scalar_t><<<blockDim, nullptr, stream>>>(x_ptr, y_ptr, z_ptr,     totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kFloat16, [&] {
                    using scalar_t = half;
                    add_kernel<scalar_t><<<blockDim, nullptr, stream>>>(x_ptr, y_ptr, z_ptr,     totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kInt32, [&] {
                    using scalar_t = int32_t;
                    add_kernel<scalar_t><<<blockDim, nullptr, stream>>>(x_ptr, y_ptr, z_ptr,     totalLength, blockLength, tileSize);
                })
            );
            return 0;
        };
        // RunOpApi/RunOpApiV2 must be used for the invocation to ensure that the timing is consistent with that of aclnn API calls made by TorchNPU.
        at_npu::native::OpCommand::RunOpApi("Add", acl_call);
        return z;
    }

    /**
     * Register the invocation function of the operator with the framework, where Device is PrivateUse1
     * The framework knows that when all inputs are on the NPU device, it dispatches to this operator implementation
     */
    // Register the NPU implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
    {
        m.impl("add", add_npu);
    }

    }  // namespace Add
    }  // namespace ascend_ops

    ```
4. Build the wheel package by referring to the [Installation Steps](#installation-steps) section, and then install and test it.
5. For testing the operator API, refer to the implementation of [test_add.py](tests/add/test_add.py)
