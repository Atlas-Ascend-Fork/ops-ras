/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <iostream>

#include "acl/acl.h"
#include "aclnnop/aclnn_matmul_abft_verify.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...) \
    do {                        \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

using Bf16 = uint16_t;

Bf16 FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::copy_n(reinterpret_cast<const unsigned char *>(&value), sizeof(value),
        reinterpret_cast<unsigned char *>(&bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<Bf16>(bits >> 16U);
}

float Bf16ToFloat(Bf16 value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::copy_n(reinterpret_cast<const unsigned char *>(&bits), sizeof(bits),
        reinterpret_cast<unsigned char *>(&result));
    return result;
}

struct TestArgs {
    int64_t m = 4096;
    int64_t n = 4096;
    int64_t k = 4096;
    int32_t deviceId = 0;
    double eMax = 0.001;
    int32_t verbose = 1;

    void Print() const
    {
        LOG_PRINT("Args: m=%ld n=%ld k=%ld eMax=%f verbose=%d deviceId=%d\n",
                  m, n, k, eMax, verbose, deviceId);
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename T>
int CopyDeviceToHost(std::vector<T>& hostData, const void* deviceAddr, const char* name)
{
    const size_t bytes = hostData.size() * sizeof(T);
    const auto ret = aclrtMemcpy(hostData.data(), bytes, deviceAddr, bytes,
                                 ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name, ret);
              return ret);
    return ACL_SUCCESS;
}


int InitAcl(int32_t deviceId, aclrtContext* context, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void PrintFloatChecksum(const std::vector<int64_t>& shape, void* deviceAddr, const char* name)
{
    std::vector<float> hostData(GetShapeSize(shape), 0.0F);
    if (CopyDeviceToHost(hostData, deviceAddr, name) != ACL_SUCCESS) {
        return;
    }
    double sum = 0.0;
    for (auto value : hostData) {
        sum += static_cast<double>(value);
    }
    LOG_PRINT("%s (FP32) first 8:", name);
    for (int64_t i = 0; i < std::min<int64_t>(hostData.size(), 8); ++i) {
        LOG_PRINT(" %.6f", hostData[i]);
    }
    LOG_PRINT("  checksum: %.6f\n", sum);
}

void PrintBf16Checksum(const std::vector<int64_t>& shape, void* deviceAddr, const char* name)
{
    std::vector<Bf16> hostData(GetShapeSize(shape), 0);
    if (CopyDeviceToHost(hostData, deviceAddr, name) != ACL_SUCCESS) {
        return;
    }
    double sum = 0.0;
    for (auto value : hostData) {
        sum += static_cast<double>(Bf16ToFloat(value));
    }
    LOG_PRINT("%s (BF16) first 8:", name);
    for (int64_t i = 0; i < std::min<int64_t>(hostData.size(), 8); ++i) {
        LOG_PRINT(" %.6f", Bf16ToFloat(hostData[i]));
    }
    LOG_PRINT("  checksum: %.6f\n", sum);
}

void PrintUint8Checksum(const std::vector<int64_t>& shape, void* deviceAddr, const char* name)
{
    std::vector<uint8_t> hostData(GetShapeSize(shape), 0);
    if (CopyDeviceToHost(hostData, deviceAddr, name) != ACL_SUCCESS) {
        return;
    }
    uint64_t sum = 0;
    for (auto value : hostData) {
        sum += static_cast<uint64_t>(value);
    }
    LOG_PRINT("%s (U8) first 8:", name);
    for (int64_t i = 0; i < std::min<int64_t>(hostData.size(), 8); ++i) {
        LOG_PRINT(" %u", static_cast<uint32_t>(hostData[i]));
    }
    LOG_PRINT("  checksum: %lu\n", sum);
}

void PrintChecksum(const std::vector<int64_t>& shape, void* deviceAddr, aclDataType dataType, const char* name)
{
    if (dataType == ACL_FLOAT) {
        PrintFloatChecksum(shape, deviceAddr, name);
    } else if (dataType == ACL_BF16) {
        PrintBf16Checksum(shape, deviceAddr, name);
    } else if (dataType == ACL_UINT8) {
        PrintUint8Checksum(shape, deviceAddr, name);
    }
}

struct ExampleShapes {
    std::vector<int64_t> a;
    std::vector<int64_t> b;
    std::vector<int64_t> xv;
    std::vector<int64_t> c;
    std::vector<int64_t> compRow;
};

ExampleShapes CreateExampleShapes(const TestArgs& args)
{
    const int64_t splitN = (args.n + 255) / 256;
    return {{args.m, args.k}, {args.k, args.n}, {args.n}, {args.m, args.n},
            {((args.m + 7) / 8) * splitN}};
}

struct ExampleData {
    std::vector<Bf16> a;
    std::vector<Bf16> b;
    std::vector<Bf16> xv;
    std::vector<float> c;
    std::vector<uint8_t> compRow;
};

ExampleData CreateExampleData(const ExampleShapes& shapes)
{
    ExampleData data;
    data.a.assign(GetShapeSize(shapes.a), 0);
    data.b.assign(GetShapeSize(shapes.b), 0);
    data.xv.assign(GetShapeSize(shapes.xv), 0);
    data.c.assign(GetShapeSize(shapes.c), 0.0F);
    data.compRow.assign(GetShapeSize(shapes.compRow), 0);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& value : data.a) {
        value = FloatToBf16(dist(rng));
    }
    for (auto& value : data.b) {
        value = FloatToBf16(dist(rng));
    }
    for (auto& value : data.xv) {
        value = FloatToBf16(1.0F);
    }
    return data;
}

struct ExampleTensors {
    void *devA = nullptr, *devB = nullptr, *devXv = nullptr, *devC = nullptr, *devCompRow = nullptr;
    aclTensor *a = nullptr, *b = nullptr, *xv = nullptr, *c = nullptr, *compRow = nullptr;
};

int CreateExampleTensors(const ExampleData& data, const ExampleShapes& shapes, ExampleTensors* tensors)
{
    auto ret = CreateAclTensor(data.a, shapes.a, &tensors->devA, ACL_BF16, &tensors->a);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(data.b, shapes.b, &tensors->devB, ACL_BF16, &tensors->b);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(data.xv, shapes.xv, &tensors->devXv, ACL_BF16, &tensors->xv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(data.c, shapes.c, &tensors->devC, ACL_FLOAT, &tensors->c);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return CreateAclTensor(data.compRow, shapes.compRow, &tensors->devCompRow, ACL_UINT8, &tensors->compRow);
}

void DestroyExampleTensors(ExampleTensors* tensors)
{
    auto destroy = [](aclTensor*& tensor) { if (tensor) { aclDestroyTensor(tensor); tensor = nullptr; } };
    auto free = [](void*& address) { if (address) { aclrtFree(address); address = nullptr; } };
    destroy(tensors->a); destroy(tensors->b); destroy(tensors->xv); destroy(tensors->c); destroy(tensors->compRow);
    free(tensors->devA); free(tensors->devB); free(tensors->devXv); free(tensors->devC); free(tensors->devCompRow);
}

int ExecuteMatmulAbftVerify(const TestArgs& args, const ExampleTensors& tensors, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMatmulAbftVerifyGetWorkspaceSize(tensors.a, tensors.b, tensors.c, tensors.xv, args.eMax,
                                                     tensors.compRow, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulAbftVerifyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnMatmulAbftVerify(workspaceAddr, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    aclrtFree(workspaceAddr);
    aclDestroyAclOpExecutor(executor);
    return ret;
}

int RunMatmulAbftVerify(const TestArgs& args, aclrtStream stream)
{
    const auto shapes = CreateExampleShapes(args);
    const auto data = CreateExampleData(shapes);
    ExampleTensors tensors;
    auto ret = CreateExampleTensors(data, shapes, &tensors);
    if (ret == ACL_SUCCESS) {
        ret = ExecuteMatmulAbftVerify(args, tensors, stream);
    }
    if (ret == ACL_SUCCESS && args.verbose) {
        PrintChecksum(shapes.c, tensors.devC, ACL_FLOAT, "c");
        PrintChecksum(shapes.compRow, tensors.devCompRow, ACL_UINT8, "compRow");
    }
    DestroyExampleTensors(&tensors);
    return ret;
}

void FinalizeAcl(int32_t deviceId, aclrtContext context, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main()
{
    TestArgs args;
    args.Print();

    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(args.deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    ret = RunMatmulAbftVerify(args, stream);
    FinalizeAcl(args.deviceId, context, stream);
    return ret;
}
