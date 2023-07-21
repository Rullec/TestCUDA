#ifndef CUBLAS_UTIL_H_
#define CUBLAS_UTIL_H_
#include "utils/EigenUtil.h"
#include <cublas_v2.h>

class cCublasUtil
{
public:
    static cublasStatus_t eigenToCublas(const tMatrixXf &eigenMat, float **devPtr);
    static cublasStatus_t cublasToEigen(const float *devPtr, tMatrixXf &eigenMat);
    static cublasStatus_t eigenVectorToCublas(const tVectorXf &eigenVec,
                                       float **devPtr);

    static cublasStatus_t cublasVectorToEigen(int size, const float *devPtr,
                                       tVectorXf &eigenVec);

    static void matrixVectorMultiply(cublasHandle_t handle, float *d_A, float *d_x,
                              float *d_y, int m, int n);
    static void ReleaseCudaMem(void * devPtr);
};

#endif
