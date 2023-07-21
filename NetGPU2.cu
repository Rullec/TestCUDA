#include "NetGPU.cuh"
#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaDef.h"
#include "gpu_utils/CudaIntrinsic.h"
#include <vector>
__device__ __inline__ void BLAS_MATMUL_AB(const float *A, int A_rows,
                                          int A_cols, const float *B,
                                          int B_cols, float *sol, int debug)
{
    if (debug != 0)
        printf("begin blas\n");

    for (int i = 0; i < A_rows; i++)
    {
        if (debug != 0)
            printf("begin blas row %d\n", i);

        for (int j = 0; j < B_cols; j++)
        {
            // sol: (A_rows, B_cols)
            sol[j * A_rows + i] = 0;
            for (int k = 0; k < A_cols; k++)
            {
                // if (debug != 0)
                //     printf("A(%d, %d) = %.3f, B(%d, %d) = %.3f\n", i, k,
                //            A[i * A_cols + k], k, j, B[k * B_cols + j]);

                sol[j * A_rows + i] += A[k * A_rows + i] * B[j * A_cols + k];
            }
            if (debug != 0)
            {
                printf("sol[%d, %d] = %.3f\n", i, j, sol[j * A_rows + i]);
            }
        }
    }
    if (debug != 0)
        printf("done blas\n");
}

__device__ void DiagMatmul_with_softplus_z(const float *z, float *cur_multi_buf,
                                           int w_rows, int w_cols)
{
    for (int i = 0; i < w_rows; i++)
    {
        float act_grad = SoftplusGrad(z[i]);
        for (int j = 0; j < w_cols; j++)
        {
            // for i-th row, scale it
            cur_multi_buf[j * w_rows + i] =
                cur_multi_buf[j * w_rows + i] * act_grad;
        }
    }
}
template <int num>
__device__ void NetworkForward_energy_grad(
    int tid, const tCudaMatrix<float, num, 1> &x_cur_, int output_dim,
    size_t num_mid_layers, devPtr<const uint> layer_arr,
    devPtr<float *const> mWLst_cublas_dev,
    devPtr<float *const> mbLst_cublas_dev, devPtr<const float> input_mean,
    devPtr<const float> input_std, float output_mean, float output_std,
    // solution
    float *energy_arr, float *dEdx_arr)
{
    constexpr int num_of_buf_energy_per_T = 100;
    constexpr int sizeof_cur_multi_buf = 1100;
    constexpr int sizeof_cur_result_buf0 = 1100;
    constexpr int sizeof_cur_result_buf1 = 100;
    float cur_triangle_st_comp_buf[num_of_buf_energy_per_T];
    float cur_multi_buf[sizeof_cur_multi_buf];
    float cur_result_buf0[sizeof_cur_result_buf0];
    float cur_result_buf1[sizeof_cur_result_buf1];

    int cur_comp_buf_offset = 0;

    using tVecType = tCudaMatrix<float, num, 1>;
    tVecType x_cur = x_cur_;
    Normalize<num>(x_cur, input_mean, input_std);

    // if (tid == 0)
    {
        // printf("-------------- CUDA kernel begin ------------\n");
        // // mInputMeanGPU[0];
        // printf("t %ld, x(after normalized) = ", tid);
        // PrintCublasVec(num, x_cur.mData);
        // 1. show input mean, std

        // printf("input mean %.3f std %.3f\n", mInputMeanGPU[0],
        // mInputStdGPU[0]);
        // printf("num layers = %ld, layer = ", num_mid_layers);
        // for (size_t i = 0; i < num_mid_layers; i++)
        // printf(" %d", layer_arr[i]);
        // printf("\n");
        int total_layers = num_mid_layers + 1;
        // printf("w.b shape = \n");
        int prev_rows = 0;
        int num_of_row_cur_reuslt = 0, num_of_col_cur_reuslt = 0;
        for (size_t layer_id = 0; layer_id < total_layers; layer_id++)
        {
            int w_rows, w_cols, b_size;
            GetWbShape(layer_arr, num_mid_layers, num, output_dim, layer_id,
                       w_rows, w_cols, b_size);

            // if (layer_id == 0)
            // {
            //     printf("w = \n");
            //     PrintCublasMat(w_rows, w_cols, mWLst_cublas_dev[layer_id]);

            //     printf("b = \n");
            //     PrintCublasVec(w_rows, mbLst_cublas_dev[layer_id]);
            // }
            // printf("(%d,%d), %d\n", w_rows, w_cols, b_size);
            // if (layer_id == 2)
            // {
            //     printf("w = \n");
            //     PrintCublasMat(w_rows, w_cols, mWLst_cublas_dev[layer_id]);
            //     // do forward calculation, print each intermediate result
            //     // printf("x = ");
            //     printf("b = \n");
            //     PrintCublasVec(w_rows, mbLst_cublas_dev[layer_id]);
            // }
            const float *x_input = (layer_id != 0)
                                       ? (cur_triangle_st_comp_buf +
                                          cur_comp_buf_offset - prev_rows)
                                       : (x_cur.mData);
            BLAS_Ax_plus_b(mWLst_cublas_dev[layer_id], w_rows, w_cols,
                           mbLst_cublas_dev[layer_id], x_input,
                           cur_triangle_st_comp_buf + cur_comp_buf_offset, 0);
            // printf("layer %ld z(before act) = ", layer_id);
            // PrintCublasVec(w_rows,
            //                cur_triangle_st_comp_buf + cur_comp_buf_offset);
            // calculate for gradient
            {
                // 1. set up cur_multi
                assert(sizeof_cur_multi_buf > w_rows * w_cols);
                for (int i = 0; i < w_rows * w_cols; i++)
                {
                    cur_multi_buf[i] = mWLst_cublas_dev[layer_id][i];
                }

                if (layer_id != total_layers - 1)
                {
                    // for current z, calculate its act grad
                    DiagMatmul_with_softplus_z(cur_triangle_st_comp_buf +
                                                   cur_comp_buf_offset,
                                               cur_multi_buf, w_rows, w_cols);
                }

                // printf("layer %ld cur_multi = \n", layer_id);
                // PrintCublasMat(w_rows, w_cols, cur_multi_buf);

                // 2. set up current result
                if (layer_id == 0)
                {
                    for (int i = 0; i < w_rows * w_cols; i++)
                    {
                        cur_result_buf0[i] = cur_multi_buf[i];
                    }
                    assert(sizeof_cur_result_buf0 > w_rows * w_cols);
                    num_of_col_cur_reuslt = w_cols;
                    num_of_row_cur_reuslt = w_rows;
                }
                else
                {
                    BLAS_MATMUL_AB(cur_multi_buf, w_rows, w_cols,
                                   cur_result_buf0, num_of_col_cur_reuslt,
                                   cur_result_buf1, 0);
                    assert(sizeof_cur_result_buf1 >
                           w_rows * num_of_col_cur_reuslt);
                    for (int i = 0; i < w_rows * num_of_col_cur_reuslt; i++)
                    {
                        cur_result_buf0[i] = cur_result_buf1[i];
                    }
                    num_of_row_cur_reuslt = w_rows;
                }

                // printf("layer %ld cur_result = \n", layer_id);
                // PrintCublasMat(num_of_row_cur_reuslt, num_of_col_cur_reuslt,
                //                cur_result_buf0);
            }

            if (layer_id != total_layers - 1)
            {
                ApplySoftplus(cur_triangle_st_comp_buf + cur_comp_buf_offset,
                              w_rows);
                // printf("layer %ld x cur = ", layer_id);
                // PrintCublasVec(w_cols, cur_triangle_st_comp_buf +
                //                            cur_comp_buf_offset - prev_rows);

                // printf("layer %ld z(after act) = ", layer_id);
                // PrintCublasVec(w_rows,
                //                cur_triangle_st_comp_buf +
                //                cur_comp_buf_offset);

                // printf("layer %ld x next = ", layer_id);
                // PrintCublasVec(w_rows,
                //                cur_triangle_st_comp_buf +
                //                cur_comp_buf_offset);
            }

            cur_comp_buf_offset += w_rows;
            prev_rows = w_rows;
            assert(cur_comp_buf_offset < num_of_buf_energy_per_T);
        }

        // 2. print W and b matrix

        energy_arr[0] =
            (cur_triangle_st_comp_buf + cur_comp_buf_offset - prev_rows)[0] *
                output_std +
            output_mean;

        for (int i = 0; i < num; i++)
        {
            // printf("[normed] %ld, output std %.3f input std %.3f\n", i,
            //        output_std, input_std[i]);
            dEdx_arr[i] = cur_result_buf0[i] * output_std / input_std[i];
            // printf("[gpu] dEdx for t%ld = %.3f\n", tid, dEdx_arr[tid][i]);
        }
        // printf("energy %ld = %.3f\n", tid, energy_arr[tid]);
    }
}

template <int num>
__global__ void NetworkForward_energy_grad_global(
    int n_ele, devPtr<const const tCudaMatrix<float, num, 1>> x_arr,
    int output_dim, size_t num_mid_layers, devPtr<const uint> layer_arr,
    devPtr<float *const> mWLst_cublas_dev,
    devPtr<float *const> mbLst_cublas_dev, devPtr<const float> input_mean,
    devPtr<const float> input_std, float output_mean, float output_std,
    // solution
    devPtr<float> energy_arr, devPtr<tCudaMatrix<float, num, 1>> dEdx_arr)
{

    CUDA_function;

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_ele)
        return;
    float E = 0;
    float dEdx[num];
    const tCudaMatrix<float, num, 1> x_cur = x_arr[tid];
    NetworkForward_energy_grad<num>(tid, x_cur, output_dim, num_mid_layers,
                                    layer_arr, mWLst_cublas_dev,
                                    mbLst_cublas_dev, input_mean, input_std,
                                    output_mean, output_std, &E, dEdx);
    energy_arr[tid] = E;
    for (int i = 0; i < num; i++)
    {
        dEdx_arr[tid][i] = dEdx[i];
    }
}
template <int num>
__global__ void NetworkForward_energy_grad_hess_global(
    int n_ele, devPtr<const const tCudaMatrix<float, num, 1>> x_arr,
    int output_dim, size_t num_mid_layers, devPtr<const uint> layer_arr,
    devPtr<float *const> mWLst_cublas_dev,
    devPtr<float *const> mbLst_cublas_dev, devPtr<const float> input_mean,
    devPtr<const float> input_std, float output_mean, float output_std,
    // solution
    devPtr<float> energy_arr, devPtr<tCudaMatrix<float, num, 1>> dEdx_arr,
    devPtr<tCudaMatrix<float, num, num>> dE2dx2_arr)
{

    CUDA_function;

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_ele)
        return;
    float E = 0;
    float dEdx[num];
    tCudaMatrix<float, num, 1> x_cur = x_arr[tid];
    NetworkForward_energy_grad<num>(tid, x_cur, output_dim, num_mid_layers,
                                    layer_arr, mWLst_cublas_dev,
                                    mbLst_cublas_dev, input_mean, input_std,
                                    output_mean, output_std, &E, dEdx);
    energy_arr[tid] = E;
    for (int i = 0; i < num; i++)
    {
        dEdx_arr[tid][i] = dEdx[i];
    }
    // calculate hessian
    float eps = 1e-3f;
    for (int i = 0; i < num; i++)
    {
        tCudaMatrix<float, num, 1> x_add = x_arr[tid];
        x_add[i] += eps;
        float dEdx_new[num];
        NetworkForward_energy_grad<num>(tid, x_add, output_dim, num_mid_layers,
                                        layer_arr, mWLst_cublas_dev,
                                        mbLst_cublas_dev, input_mean, input_std,
                                        output_mean, output_std, &E, dEdx_new);
        for (int j = 0; j < num; j++)
        {
            // (j, i)
            dE2dx2_arr[tid](j, i) = (dEdx_new[j] - dEdx[j]) / eps;
            printf("[kernel] hess(%d %d) = %.2f\n", i, j, dE2dx2_arr[tid](j, i));
        }
    }

    for (int i = 0; i < num; i++)
        for (int j = i + 1; j < num; j++)
        {
            float mean = (dE2dx2_arr[tid](i, j) + dE2dx2_arr[tid](j, i)) / 2;

            dE2dx2_arr[tid](i, j) = mean;
            dE2dx2_arr[tid](j, i) = mean;
        }
}

#include "NetGPU.h"
void NetGPU::forward_func_1d_energy_grad(const cCudaArray<tCudaVector1f> &x_arr,
                                         cCudaArray<float> &E_arr,
                                         cCudaArray<tCudaVector1f> &dEdx_arr)
{
    int N = x_arr.Size();
    // NetworkForward_energy_grad_shared_mem CUDA_at(N, 128)(
    NetworkForward_energy_grad_global<1> CUDA_at(N, 128)(
        N, x_arr.Ptr(), 1, this->mLayersGPU.Size(), mLayersGPU.Ptr(),
        mWLst_cublas_dev.Ptr(), mbLst_cublas_dev.Ptr(), mInputMeanGPU.Ptr(),
        mInputStdGPU.Ptr(), mOutputMean, mOutputStd,

        // ============== solution ================
        E_arr.Ptr(), dEdx_arr.Ptr());
}

void NetGPU::forward_func_2d_energy_grad(const cCudaArray<tCudaVector2f> &x_arr,
                                         cCudaArray<float> &E_arr,
                                         cCudaArray<tCudaVector2f> &dEdx_arr)
{
    int N = x_arr.Size();
    // NetworkForward_energy_grad_shared_mem CUDA_at(N, 128)(
    NetworkForward_energy_grad_global<2> CUDA_at(N, 128)(
        N, x_arr.Ptr(), 1, this->mLayersGPU.Size(), mLayersGPU.Ptr(),
        mWLst_cublas_dev.Ptr(), mbLst_cublas_dev.Ptr(), mInputMeanGPU.Ptr(),
        mInputStdGPU.Ptr(), mOutputMean, mOutputStd,

        // ============== solution ================
        E_arr.Ptr(), dEdx_arr.Ptr());
}

void NetGPU::forward_func_1d_energy_grad_hess(
    const cCudaArray<tCudaVector1f> &x_arr, cCudaArray<float> &E_arr,
    cCudaArray<tCudaVector1f> &dEdx_arr, cCudaArray<tCudaMatrix1f> &dE2dx2_arr)
{
    int N = x_arr.Size();
    // NetworkForward_energy_grad_shared_mem CUDA_at(N, 128)(
    NetworkForward_energy_grad_hess_global<1> CUDA_at(N, 128)(
        N, x_arr.Ptr(), 1, this->mLayersGPU.Size(), mLayersGPU.Ptr(),
        mWLst_cublas_dev.Ptr(), mbLst_cublas_dev.Ptr(), mInputMeanGPU.Ptr(),
        mInputStdGPU.Ptr(), mOutputMean, mOutputStd,

        // ============== solution ================
        E_arr.Ptr(), dEdx_arr.Ptr(), dE2dx2_arr.Ptr());
}
void NetGPU::forward_func_2d_energy_grad_hess(
    const cCudaArray<tCudaVector2f> &x_arr, cCudaArray<float> &E_arr,
    cCudaArray<tCudaVector2f> &dEdx_arr, cCudaArray<tCudaMatrix2f> &dE2dx2_arr)
{

    int N = x_arr.Size();
    // NetworkForward_energy_grad_shared_mem CUDA_at(N, 128)(
    NetworkForward_energy_grad_hess_global<2> CUDA_at(N, 128)(
        N, x_arr.Ptr(), 1, this->mLayersGPU.Size(), mLayersGPU.Ptr(),
        mWLst_cublas_dev.Ptr(), mbLst_cublas_dev.Ptr(), mInputMeanGPU.Ptr(),
        mInputStdGPU.Ptr(), mOutputMean, mOutputStd,

        // ============== solution ================
        E_arr.Ptr(), dEdx_arr.Ptr(), dE2dx2_arr.Ptr());
}
// __global__ void NetworkForward_energy_grad_shared_mem(
//     int n_ele, devPtr<const tCudaVector1f> x_arr, int input_dim, int
//     output_dim, size_t num_mid_layers, devPtr<const uint> layer_arr,
//     devPtr<float *const> mWLst_cublas_dev_,
//     devPtr<float *const> mbLst_cublas_dev_, devPtr<const float> input_mean,
//     devPtr<const float> input_std, float output_mean, float output_std,

//     // buffer
//     devPtr<float> comp_buf_energy, int num_of_buf_energy_per_T,
//     devPtr<float> cur_multi_buf_, int sizeof_cur_multi_buf,
//     devPtr<float> cur_result_buf0_, int sizeof_cur_result_buf0,
//     devPtr<float> cur_result_buf1_, int sizeof_cur_result_buf1,
//     // solution
//     devPtr<float> energy_arr, devPtr<tCudaVector1f> dEdx_arr)
// {
//     CUDA_function;
//     int index = threadIdx.x;
//     int stride = blockDim.x;
//     size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

//     // devPtr<float> cur_triangle_st_comp_buf =
//     //     comp_buf_energy + num_of_buf_energy_per_T * tid;

//     // float *cur_multi_buf = cur_multi_buf_ + tid * sizeof_cur_multi_buf;
//     // float *cur_result_buf0 = cur_result_buf0_ + tid *
//     sizeof_cur_result_buf0;
//     // float *cur_result_buf1 = cur_result_buf1_ + tid *
//     sizeof_cur_result_buf1; float cur_triangle_st_comp_buf[100]; float
//     cur_multi_buf[1100]; float cur_result_buf0[1100]; float
//     cur_result_buf1[100];

//     __shared__ float network_param[5000];

//     int total_layers = num_mid_layers + 1;
//     int data_offset = 0;
//     constexpr int max_layer = 10;
//     int network_param_w_st_pos[max_layer]; // max 10 layers
//     int network_param_b_st_pos[max_layer]; // max 10 layers

//     for (size_t layer_id = 0; layer_id < total_layers; layer_id++)
//     {
//         int w_rows, w_cols, b_size;
//         GetWbShape(layer_arr, num_mid_layers, input_dim, output_dim,
//         layer_id,
//                    w_rows, w_cols, b_size);
//         float *w_data = mWLst_cublas_dev_[layer_id];
//         float *b_data = mbLst_cublas_dev_[layer_id];

//         // begin to copy per result
//         network_param_w_st_pos[layer_id] = data_offset;
//         for (int i = index; i < w_rows * w_cols; i += stride)
//             network_param[data_offset + i] = w_data[i];
//         data_offset += w_rows * w_cols;

//         network_param_b_st_pos[layer_id] = data_offset;
//         for (int i = index; i < b_size; i += stride)
//             network_param[data_offset + i] = b_data[i];
//         data_offset += b_size;
//     }