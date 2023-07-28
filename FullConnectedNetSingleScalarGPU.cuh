#ifndef NET_GPU_CUH_
#define NET_GPU_CUH_
#include "gpu_utils/CudaDevPtr.h"

__device__ __inline__ void GetWbShape(const uint *layer_arr, int layer_arr_size,
                                      int input_dim, int output_dim,
                                      int layer_id, int &w_rows, int &w_cols,
                                      int &b_size)
{
    if (layer_id < 0 || layer_id >= layer_arr_size + 1)
    {
        printf("[error] get wb shape failed\n");
    }
    else
    {
        if (layer_id == 0)
        {
            w_rows = layer_arr[layer_id];
            w_cols = input_dim;
        }
        else if (layer_id == layer_arr_size)
        {
            w_rows = output_dim;
            w_cols = layer_arr[layer_arr_size - 1];
        }
        else
        {
            w_rows = layer_arr[layer_id];
            w_cols = layer_arr[layer_id - 1];
        }
        b_size = w_rows;
    }
}

__device__ __inline__ void PrintCublasMat(int mat_rows, int mat_cols,
                                          float *const host_ptr)
{

    // Print the matrix
    for (int i = 0; i < mat_rows; i++)
    {
        for (int j = 0; j < mat_cols; j++)
        {
            printf("%.3f\t", host_ptr[j * mat_rows + i]);
        }
        printf("\n");
    }
}

__device__ __inline__ void PrintCublasVec(int vec_size, float *const host_ptr)
{
    // Print the matrix
    for (int j = 0; j < vec_size; j++)
    {
        printf("%.3f\t", host_ptr[j]);
    }
    printf("\n");
}

__device__ __inline__ float Softplus(const float &x)
{
    return std::log(1.0f + std::exp(x));
    // return 0.0f;
}

__device__ __inline__ float SoftplusGrad(const float &x)
{
    float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
    // return 0.0f;
}

__device__ __inline__ void ApplySoftplus(devPtr<float> data_ptr, int N)
{
    for (int i = 0; i < N; i++)
    {
        data_ptr[i] = std::log(1.0f + std::exp(data_ptr[i]));
    }
}

__device__ __inline__ void BLAS_Ax_plus_b_column_major(const float *A,
                                                       int A_rows, int A_cols,
                                                       const float *b,
                                                       const float *x,
                                                       float *sol, int debug)
{
    for (int i = 0; i < A_rows; i++)
        sol[i] = b[i];
    for (int j = 0; j < A_cols; j++)
    {
        for (int i = 0; i < A_rows; i++)
        {
            sol[i] += A[j * A_rows + i] * x[j];
        }
    }
}

__device__ __inline__ void BLAS_Ax_plus_b_row_major(const float *A, int A_rows,
                                                    int A_cols, const float *b,
                                                    const float *x, float *sol,
                                                    int debug)
{
    for (int i = 0; i < A_rows; i++)
        sol[i] = b[i];
    for (int i = 0; i < A_rows; i++)
    {
        for (int j = 0; j < A_cols; j++)
        {
            sol[i] += A[i * A_cols + j] * x[j];
        }
    }
}

template <int num>
__device__ void Normalize(tCudaMatrix<float, num, 1> &x_cur,
                          devPtr<const float> mean, devPtr<const float> std)
{
    using tVecType = tCudaMatrix<float, num, 1>;

    for (int i = 0; i < tVecType::mElements; i++)
        x_cur[i] = (x_cur[i] - mean[i]) / std[i];
}

#endif