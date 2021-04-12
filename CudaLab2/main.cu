#include <new>
#include <random>
#include <cstddef>
#include <iostream>
#include <cinttypes>
#include <algorithm>
#include <windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

static const unsigned int CUDA_BLOCK_SIZE = 32;

static const int UNIFORM_DISTR_MIN = -100;
static const int UNIFORM_DISTR_MAX = 100;

static const size_t MATRIX_SIZES_TO_TEST[3] = { 1000, 1500, 2000 };


__global__ void _multKernel(double* m1, double* m2, size_t mSize, double* res)
{
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= mSize || j >= mSize)
        return;

    size_t ind = i * mSize + j;
    res[ind] = 0;

    for (size_t k = 0; k < mSize; ++k) {
        res[ind] += m1[i * mSize + k] * m2[k * mSize + j];
    }
}


__global__ void _multKernelShared(double* m1, double* m2, size_t mSize, double* res)
{
    __shared__ double a[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
    __shared__ double b[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

    size_t ty = threadIdx.y;
    size_t tx = threadIdx.x;
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    size_t row = by * CUDA_BLOCK_SIZE + ty;
    size_t col = bx * CUDA_BLOCK_SIZE + tx;
    double sum = 0;

    for (size_t i = 0, m1_col = tx, m2_row = ty; i * CUDA_BLOCK_SIZE < mSize; ++i, m1_col += CUDA_BLOCK_SIZE, m2_row += CUDA_BLOCK_SIZE)
    {
        if (row < mSize && m1_col < mSize)
        {
            a[ty][tx] = m1[row * mSize + m1_col];
        } else {
            a[ty][tx] = 0;
        }

        if (col < mSize && m2_row < mSize)
        {
            b[ty][tx] = m2[m2_row * mSize + col];
        } else {
            b[ty][tx] = 0;
        }

        __syncthreads();
        for (size_t k = 0; k < CUDA_BLOCK_SIZE; k++) {
            sum += a[ty][k] * b[k][tx];
        }
        __syncthreads();
    }

    if (row < mSize && col < mSize) {
        res[row * mSize + col] = sum;
    }
}


static float _gpuMult(double* m1, double* m2, size_t mSize, double* res, bool isShared = false)
{
    cudaEvent_t cudaStartEvent, cudaEndEvent;
    float gpuTime = 0.0f;

    double* cudaMemM1;
    double* cudaMemM2;
    double* cudaMemRes;
    size_t matrixBytesNum = sizeof(double) * mSize * mSize;

    dim3 cudaThreads(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 cudaBlocks((mSize + cudaThreads.x - 1) / cudaThreads.x, (mSize + cudaThreads.y - 1) / cudaThreads.y);

    cudaMalloc((void**)&cudaMemM1, matrixBytesNum);
    cudaMalloc((void**)&cudaMemM2, matrixBytesNum);
    cudaMalloc((void**)&cudaMemRes, matrixBytesNum);

    cudaEventCreate(&cudaStartEvent);
    cudaEventCreate(&cudaEndEvent);

    // asynchronously issue work to the GPU (all to stream 0)
    cudaEventRecord(cudaStartEvent, 0);
    cudaMemcpy(cudaMemM1, m1, matrixBytesNum, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaMemM2, m2, matrixBytesNum, cudaMemcpyHostToDevice);

    if (!isShared) {
        _multKernel<< <cudaBlocks, cudaThreads >> >(cudaMemM1, cudaMemM2, mSize, cudaMemRes);
    } else {
        _multKernelShared<< <cudaBlocks, cudaThreads >> >(cudaMemM1, cudaMemM2, mSize, cudaMemRes);
    }

    cudaEventRecord(cudaEndEvent, 0);

    // force synchronization
    cudaEventSynchronize(cudaEndEvent);
    cudaEventElapsedTime(&gpuTime, cudaStartEvent, cudaEndEvent);

    // get data back only after getting elapsed time!
    cudaMemcpy(res, cudaMemRes, matrixBytesNum, cudaMemcpyDeviceToHost);

    cudaEventDestroy(cudaStartEvent);
    cudaEventDestroy(cudaEndEvent);
    cudaFree(cudaMemM1);
    cudaFree(cudaMemM2);
    cudaFree(cudaMemRes);

    return gpuTime / 1000.0f;
}


static double _getDeviation(double* m1, double* m2, size_t mSize)
{
    size_t matrixElemsCount = mSize * mSize;
    double res = 0.0;

    for (size_t i = 0; i < matrixElemsCount; ++i) {
        res = std::max(res, std::abs(m1[i] - m2[i]));
    }

    return res;
}


static double* _getRandomMatrix(size_t matrixSize)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(UNIFORM_DISTR_MIN, UNIFORM_DISTR_MAX);

    size_t matrixElemsCount = matrixSize * matrixSize;
    double* res = new double[matrixElemsCount];

    for (size_t i = 0; i < matrixElemsCount; ++i) {
        res[i] = distrib(gen);
    }

    return res;
}


int main(int argc, char* argv[])
{
    for (size_t matrixSize : MATRIX_SIZES_TO_TEST) {
        std::cout << "=================================================" << std::endl;
        std::cout << "Matrix size: " << matrixSize << "x" << matrixSize << std::endl;

        double* m1 = _getRandomMatrix(matrixSize);
        double* m2 = _getRandomMatrix(matrixSize);

        double* resGPU = new double[matrixSize * matrixSize];
        double* resGPUShared = new double[matrixSize * matrixSize];

        float timeGPU = _gpuMult(m1, m2, matrixSize, resGPU);
        float timeGPUShared = _gpuMult(m1, m2, matrixSize, resGPUShared, true);

        std::cout << "Elapsed time on GPU: " << timeGPU << " s" << std::endl;
        std::cout << "Elapsed time on GPU with shared memory: " << timeGPUShared << " s" << std::endl;
        std::cout << "Maximum deviation: " << _getDeviation(resGPU, resGPUShared, matrixSize) << std::endl;

        delete[] m1;
        delete[] m2;
        delete[] resGPU;
        delete[] resGPUShared;
    }

    return 0;
}
