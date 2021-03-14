#include <new>
#include <random>
#include <cstddef>
#include <iostream>
#include <cinttypes>
#include <algorithm>
#include <windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static const unsigned int CUDA_BLOCK_SIZE = 32;

static const int UNIFORM_DISTR_MIN = -100;
static const int UNIFORM_DISTR_MAX =  100;

static const size_t MATRIX_SIZES_TO_TEST[3] = { 1000, 1500, 2000 };


static float _cpuMult(double * m1, double * m2, size_t mSize, double * res)
{
    LARGE_INTEGER perfStart, perfEnd, freq;
    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&perfStart);
    for (size_t i = 0; i < mSize; ++i) {
        for (size_t j = 0; j < mSize; ++j) {
            size_t ind = i * mSize + j;
            res[ind] = 0;

            for (size_t k = 0; k < mSize; ++k) {
                res[ind] += m1[i * mSize + k] * m2[k * mSize + j];
            }
        }
    }
    QueryPerformanceCounter(&perfEnd);

    return static_cast<float>(perfEnd.QuadPart - perfStart.QuadPart) / freq.QuadPart;
}


__global__ void _multKernel(double * m1, double * m2, size_t mSize, double * res)
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


static float _gpuMult(double * m1, double * m2, size_t mSize, double * res)
{
    cudaEvent_t cudaStartEvent, cudaEndEvent;
    float gpuTime = 0.0f;

    double * cudaMemM1;
    double * cudaMemM2;
    double * cudaMemRes;
    size_t matrixBytesNum = sizeof(double) * mSize * mSize;

    dim3 cudaThreads(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 cudaBlocks((mSize + cudaThreads.x - 1) / cudaThreads.x, (mSize + cudaThreads.y - 1) / cudaThreads.y);

    cudaMalloc(reinterpret_cast<void**>(&cudaMemM1), matrixBytesNum);
    cudaMalloc(reinterpret_cast<void**>(&cudaMemM2), matrixBytesNum);
    cudaMalloc(reinterpret_cast<void**>(&cudaMemRes), matrixBytesNum);

    cudaEventCreate(&cudaStartEvent);
    cudaEventCreate(&cudaEndEvent);

    // asynchronously issue work to the GPU (all to stream 0)
    cudaEventRecord(cudaStartEvent, 0);
    cudaMemcpy(cudaMemM1, m1, matrixBytesNum, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaMemM2, m2, matrixBytesNum, cudaMemcpyHostToDevice);

    _multKernel<<<cudaBlocks, cudaThreads>>>(cudaMemM1, cudaMemM2, mSize, cudaMemRes);

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


static double _getDeviation(double * m1, double * m2, size_t mSize)
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

        double * m1 = _getRandomMatrix(matrixSize);
        double * m2 = _getRandomMatrix(matrixSize);

        double * resOnCPU = new double[matrixSize * matrixSize];
        double * resOnGPU = new double[matrixSize * matrixSize];

        float timeOnCPU = _cpuMult(m1, m2, matrixSize, resOnCPU);
        float timeOnGPU = _gpuMult(m1, m2, matrixSize, resOnGPU);

        std::cout << "Elapsed time on CPU: " << timeOnCPU << " s" << std::endl;
        std::cout << "Elapsed time on GPU: " << timeOnGPU << " s" << std::endl;
        std::cout << "Maximum deviation: " << _getDeviation(resOnCPU, resOnGPU, matrixSize) << std::endl;

        delete[] m1;
        delete[] m2;
        delete[] resOnCPU;
        delete[] resOnGPU;
    }

    return 0;
}
