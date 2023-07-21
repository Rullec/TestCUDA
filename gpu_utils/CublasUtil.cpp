#include "gpu_utils/CublasUtil.h"
#include <cuda_runtime_api.h>

cublasStatus_t cCublasUtil::eigenToCublas(const Eigen::MatrixXf &eigenMat,
                                          float **devPtr)
{
    int rows = eigenMat.rows();
    int cols = eigenMat.cols();

    const float *hostPtr = eigenMat.data();

    cublasStatus_t status;

    // Allocate memory on the device
    cudaError_t cudaStatus =
        cudaMalloc((void **)devPtr, rows * cols * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    // Copy the matrix to the device
    status = cublasSetMatrix(rows, cols, sizeof(float), hostPtr, rows, *devPtr,
                             rows);

    return status;
}

cublasStatus_t cCublasUtil::cublasToEigen(const float *devPtr,
                                          Eigen::MatrixXf &eigenMat)
{
    cublasStatus_t status;
    int rows = eigenMat.rows(), cols = eigenMat.cols();
    float *hostPtr = eigenMat.data();

    // Copy the matrix from the device to the host
    status =
        cublasGetMatrix(rows, cols, sizeof(float), devPtr, rows, hostPtr, rows);

    return status;
}

cublasStatus_t cCublasUtil::eigenVectorToCublas(const Eigen::VectorXf &eigenVec,
                                                float **devPtr)
{
    int size = eigenVec.size();

    const float *hostPtr = eigenVec.data();

    cublasStatus_t status;

    // Allocate memory on the device
    cudaError_t cudaStatus = cudaMalloc((void **)devPtr, size * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }

    // Copy the vector to the device
    status = cublasSetVector(size, sizeof(float), hostPtr, 1, *devPtr, 1);

    return status;
}

cublasStatus_t cCublasUtil::cublasVectorToEigen(int size, const float *devPtr,
                                                Eigen::VectorXf &eigenVec)
{
    cublasStatus_t status;

    float *hostPtr = eigenVec.data();

    // Copy the vector from the device to the host
    status = cublasGetVector(size, sizeof(float), devPtr, 1, hostPtr, 1);

    return status;
}

void cCublasUtil::matrixVectorMultiply(cublasHandle_t handle, float *d_A,
                                       float *d_x, float *d_y, int m, int n)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_y,
                1);
}

void cCublasUtil::ReleaseCudaMem(void *devPtr) { cudaFree(devPtr); }