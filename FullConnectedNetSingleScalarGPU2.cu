#include "FullConnectedNetSingleScalarGPU.cuh"
#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaDef.h"
#include "gpu_utils/CudaIntrinsic.h"
#include <vector>
__device__ __inline__ void BLAS_MATMUL_AB(const float *A, int A_rows,
                                          int A_cols, const float *B,
                                          int B_cols, float *sol, int debug)
{
    // if (debug != 0)
    //     printf("begin blas\n");

    for (int i = 0; i < A_rows; i++)
    {
        // if (debug != 0)
        //     printf("begin blas row %d\n", i);

        for (int j = 0; j < B_cols; j++)
        {
            // sol: (A_rows, B_cols)
            sol[j * A_rows + i] = 0.0f;
            for (int k = 0; k < A_cols; k++)
            {
                // if (debug != 0)
                //     printf("A(%d, %d) = %.3f, B(%d, %d) = %.3f\n", i, k,
                //            A[i * A_cols + k], k, j, B[k * B_cols + j]);

                sol[j * A_rows + i] += A[k * A_rows + i] * B[j * A_cols + k];
            }
            // if (debug != 0)
            // {
            //     printf("sol[%d, %d] = %.3f\n", i, j, sol[j * A_rows + i]);
            // }
        }
    }
    // if (debug != 0)
    //     printf("done blas\n");
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
    size_t num_mid_layers, const uint *layer_arr,
    devPtr<float *const> mWLst_cublas_dev,
    devPtr<float *const> mbLst_cublas_dev, devPtr<const float> input_mean,
    devPtr<const float> input_std, float output_mean, float output_std,
    const float *net_param_sm, int *w_st_pos_arr, int *b_st_pos_arr,
    // solution
    float *energy_arr, float *dEdx_arr)
{
    // constexpr int num_of_buf_energy_per_T = 100;
    // constexpr int sizeof_cur_multi_buf = 1100;
    // constexpr int sizeof_cur_result_buf0 = 1100;
    // constexpr int sizeof_cur_result_buf1 = 100;
    // float cur_multi_buf[sizeof_cur_multi_buf];
    // float cur_result_buf0[sizeof_cur_result_buf0];
    // float cur_result_buf1[sizeof_cur_result_buf1];

    constexpr int max_net_width = 32;
    float energy_forward_buf[max_net_width * 2];
    bool energy_buf_second_seg_used = 0;
    constexpr int max_jac_size = max_net_width * num;
    float jac_forward_buf[max_jac_size * num];
    bool jac_buf_second_seg_used = 0;

    // printf("[kernel] begin\n");
    int cur_comp_buf_offset = 0;

    using tVecType = tCudaMatrix<float, num, 1>;
    tVecType x_cur = x_cur_;
    Normalize<num>(x_cur, input_mean, input_std);

    // copy x to energy_forward_buf
#pragma unroll
    for (int i = 0; i < num; i++)
        energy_forward_buf[i] = x_cur[i];

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
        // int prev_rows = 0;
        int num_of_row_cur_reuslt = 0, num_of_col_cur_reuslt = 0;
        for (size_t layer_id = 0; layer_id < total_layers; layer_id++)
        {
            int w_rows, w_cols, b_size;
            GetWbShape(layer_arr, num_mid_layers, num, output_dim, layer_id,
                       w_rows, w_cols, b_size);
            const float *w_data = net_param_sm + w_st_pos_arr[layer_id];
            const float *b_data = net_param_sm + b_st_pos_arr[layer_id];
            const float *x_input =
                energy_forward_buf + max_net_width * energy_buf_second_seg_used;
            float *x_output =
                energy_forward_buf +
                max_net_width * (false == energy_buf_second_seg_used);
            // BLAS_Ax_plus_b_column_major(mWLst_cublas_dev[layer_id], w_rows,
            // w_cols,
            //                mbLst_cublas_dev[layer_id], x_input, x_output, 0);
            BLAS_Ax_plus_b_column_major(w_data, w_rows, w_cols, b_data, x_input,
                                        x_output, 0);
            energy_buf_second_seg_used = !energy_buf_second_seg_used;

            // printf("layer %ld z(before act) = ", layer_id);
            // PrintCublasVec(w_rows,
            //                cur_triangle_st_comp_buf + cur_comp_buf_offset);
            // calculate for gradient
            // printf("layer %d jac begin\n", layer_id);
            {
                const float *jac_input =
                    jac_forward_buf + max_jac_size * jac_buf_second_seg_used;
                float *jac_output =
                    jac_forward_buf +
                    max_jac_size * (false == jac_buf_second_seg_used);
                jac_buf_second_seg_used = !jac_buf_second_seg_used;

                // 1. set up cur_multi

                // printf("layer %d jac confirmed\n", layer_id);
                // copy first w to jac_forward_buf[0], scale each row as act z
                if (layer_id == 0)
                {
                    // printf("layer %d jac calc_type0 begin\n", layer_id);
#pragma unroll
                    for (int j = 0; j < num; j++)
                    {
                        for (int i = 0; i < w_rows; i++)
                        {
                            float val = SoftplusGrad(x_output[i]);
                            jac_output[j * w_rows + i] =
                                val * w_data[j * w_rows + i];
                        }
                    }
                    // printf("layer %d jac calc_type0 succ\n", layer_id);
                }
                else if (layer_id != total_layers - 1)
                {
// jac_input: (w_cols, num)
// jac_output(w_rows, num) = scaled_w(w_rows, w_cols) *
// (w_cols, num)
#pragma once
                    for (int j = 0; j < num; j++)
                    {
                        // printf("layer %d jac calc_type1 begin\n", layer_id);
                        // printf("layer %d jac calc_type1 begin loop\n",
                        //        layer_id);
                        for (int i = 0; i < w_rows; i++)
                        {
                            float val = SoftplusGrad(x_output[i]);
                            // col_id * num_rows + row_id
                            jac_output[j * w_rows + i] = 0.0f;
                            for (int k = 0; k < w_cols; k++)
                            {
                                jac_output[j * w_rows + i] +=
                                    w_data[k * w_rows + i] *
                                    jac_input[j * w_cols + k];
                            }
                            jac_output[j * w_rows + i] *= val;
                        }
                    }
                    // printf("layer %d jac calc_type1 succ\n", layer_id);
                }
                else
                {
// printf("layer %d jac calc_type2 begin\n", layer_id);
#pragma once
                    for (int j = 0; j < num; j++)
                    {
                        // printf("layer %d jac calc_type2 begin\n", layer_id);
                        // float val = 1.0;
                        // printf("layer %d jac calc_type2 begin loop\n",
                        //        layer_id);
                        for (int i = 0; i < w_rows; i++)
                        {
                            // col_id * num_rows + row_id
                            jac_output[j * w_rows + i] = 0;
                            for (int k = 0; k < w_cols; k++)
                            {
                                jac_output[j * w_rows + i] +=
                                    w_data[k * w_rows + i] *
                                    jac_input[j * w_cols + k];
                            }
                            // jac_output[j * w_rows + i] *= val;
                        }
                    }
                    // printf("layer %d jac calc_type2 end\n", layer_id);
                }
            }

            if (layer_id != total_layers - 1)
            {
                ApplySoftplus(x_output, w_rows);
            }

            // cur_comp_buf_offset += w_rows;
            // prev_rows = w_rows;
            // assert(cur_comp_buf_offset < num_of_buf_energy_per_T);
        }

        const float *jac_input_new =
            jac_forward_buf + max_jac_size * jac_buf_second_seg_used;
        // 2. print W and b matrix

        float *x_input_new =
            energy_forward_buf + max_net_width * energy_buf_second_seg_used;
        energy_arr[0] = x_input_new[0] * output_std + output_mean;

        for (int i = 0; i < num; i++)
        {
            // printf("[normed] %ld, output std %.3f input std %.3f\n", i,
            //        output_std, input_std[i]);
            dEdx_arr[i] = jac_input_new[i] * output_std / input_std[i];
            // printf("[gpu] dEdx for t%ld = %.3f\n", tid, dEdx_arr[tid][i]);
        }
        // printf("energy %ld = %.3f\n", tid, energy_arr[tid]);
    }
}

template <int num>
__device__ void NetworkForward_energy_grad_row_major(
    int tid, const tCudaMatrix<float, num, 1> &x_cur_, int output_dim,
    size_t num_mid_layers, const uint *layer_arr,
    devPtr<const float> input_mean, devPtr<const float> input_std,
    float output_mean, float output_std, const float *net_param_sm,
    int *w_rowmajor_st_pos_arr, int *b_st_pos_arr,
    // solution
    float *energy_arr, float *dEdx_arr)
{
    // constexpr int num_of_buf_energy_per_T = 100;
    // constexpr int sizeof_cur_multi_buf = 1100;
    // constexpr int sizeof_cur_result_buf0 = 1100;
    // constexpr int sizeof_cur_result_buf1 = 100;
    // float cur_multi_buf[sizeof_cur_multi_buf];
    // float cur_result_buf0[sizeof_cur_result_buf0];
    // float cur_result_buf1[sizeof_cur_result_buf1];
    // printf("[device] forward 1 begin\n");
    constexpr int max_net_width = 32;
    float energy_forward_buf[max_net_width * 2];
    bool energy_buf_second_seg_used = 0;
    constexpr int max_jac_size = max_net_width * num;
    float jac_forward_buf[max_jac_size * 2];
    bool jac_buf_second_seg_used = 0;

    // printf("[kernel] begin\n");
    int cur_comp_buf_offset = 0;

    using tVecType = tCudaMatrix<float, num, 1>;
    tVecType x_cur = x_cur_;
    Normalize<num>(x_cur, input_mean, input_std);

    // copy x to energy_forward_buf
#pragma unroll
    for (int i = 0; i < num; i++)
        energy_forward_buf[i] = x_cur[i];

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
        // int prev_rows = 0;
        int num_of_row_cur_reuslt = 0, num_of_col_cur_reuslt = 0;
        for (size_t layer_id = 0; layer_id < total_layers; layer_id++)
        {
            // printf("[device] forward layer %d begin\n", layer_id);
            int w_rows, w_cols, b_size;
            GetWbShape(layer_arr, num_mid_layers, num, output_dim, layer_id,
                       w_rows, w_cols, b_size);
            const float *w_rowmajor_data =
                net_param_sm + w_rowmajor_st_pos_arr[layer_id];
            const float *b_data = net_param_sm + b_st_pos_arr[layer_id];
            const float *x_input =
                energy_forward_buf + max_net_width * energy_buf_second_seg_used;
            float *x_output =
                energy_forward_buf +
                max_net_width * (false == energy_buf_second_seg_used);
            // BLAS_Ax_plus_b_column_major(mWLst_cublas_dev[layer_id], w_rows,
            // w_cols,
            //                mbLst_cublas_dev[layer_id], x_input, x_output, 0);
            // printf("[device] forward layer %d BLAS begin\n", layer_id);
            BLAS_Ax_plus_b_row_major(w_rowmajor_data, w_rows, w_cols, b_data,
                                     x_input, x_output, 0);
            energy_buf_second_seg_used = !energy_buf_second_seg_used;

            // printf("layer %ld z(before act) = ", layer_id);
            // PrintCublasVec(w_rows,
            //                cur_triangle_st_comp_buf + cur_comp_buf_offset);
            // calculate for gradient
            // printf("layer %d jac begin\n", layer_id);
            // printf("[device] forward layer %d jac begin\n", layer_id);
            {
                const float *jac_input_column_major =
                    jac_forward_buf + max_jac_size * jac_buf_second_seg_used;
                float *jac_output_column_major =
                    jac_forward_buf +
                    max_jac_size * (false == jac_buf_second_seg_used);
                jac_buf_second_seg_used = !jac_buf_second_seg_used;

                // 1. set up cur_multi

                // printf("layer %d jac confirmed\n", layer_id);
                // copy first w to jac_forward_buf[0], scale each row as act z
                if (layer_id == 0)
                {
                    // printf("layer %d jac calc_type0 begin\n", layer_id);
                    for (int i = 0; i < w_rows; i++)
                    {
#pragma unroll
                        for (int j = 0; j < num; j++)
                        {
                            float val = SoftplusGrad(x_output[i]);
                            jac_output_column_major[j * w_rows + i] =
                                val * w_rowmajor_data[i * num + j];
                        }
                    }
                    // printf("layer %d jac calc_type0 succ\n", layer_id);
                }
                else if (layer_id != total_layers - 1)
                {
                    // jac_input: (w_cols, num)
                    // jac_output(w_rows, num) = scaled_w(w_rows, w_cols) *
                    // (w_cols, num)

                    for (int row_id = 0; row_id < w_rows; row_id++)
                    {
                        float val = SoftplusGrad(x_output[row_id]);
#pragma unroll
                        for (int col_id = 0; col_id < num; col_id++)
                        {
                            // jac_output_column_major, shape (rows_, num)
                            jac_output_column_major[col_id * w_rows + row_id] =
                                0.0f;

                            for (int k = 0; k < w_cols; k++)
                            {
                                // w(row_id, k), row major
                                // jac(k, col_id), column major
                                jac_output_column_major[col_id * w_rows +
                                                        row_id] +=
                                    val * w_rowmajor_data[row_id * w_cols + k] *
                                    jac_input_column_major[col_id * w_cols + k];
                            }
                        }
                    }

                    // printf("layer %d jac calc_type1 succ\n", layer_id);
                }
                else
                {
                    for (int row_id = 0; row_id < w_rows; row_id++)
                    {
#pragma unroll
                        for (int col_id = 0; col_id < num; col_id++)
                        {
                            // jac_output_column_major, shape (rows_, num)
                            jac_output_column_major[col_id * w_rows + row_id] =
                                0.0f;

                            for (int k = 0; k < w_cols; k++)
                            {
                                // w(row_id, k), row major
                                // jac(k, col_id), column major
                                jac_output_column_major[col_id * w_rows +
                                                        row_id] +=
                                    w_rowmajor_data[row_id * w_cols + k] *
                                    jac_input_column_major[col_id * w_cols + k];
                            }
                        }
                    }
                    // printf("layer %d jac calc_type2 end\n", layer_id);
                }
            }
            // printf("[device] forward layer %d softplus begin\n", layer_id);
            if (layer_id != total_layers - 1)
            {
                ApplySoftplus(x_output, w_rows);
            }

            // cur_comp_buf_offset += w_rows;
            // prev_rows = w_rows;
            // assert(cur_comp_buf_offset < num_of_buf_energy_per_T);
        }

        const float *jac_input_new =
            jac_forward_buf + max_jac_size * jac_buf_second_seg_used;
        // 2. print W and b matrix

        float *x_input_new =
            energy_forward_buf + max_net_width * energy_buf_second_seg_used;
        energy_arr[0] = x_input_new[0] * output_std + output_mean;

        for (int i = 0; i < num; i++)
        {
            // printf("[normed] %ld, output std %.3f input std %.3f\n", i,
            //        output_std, input_std[i]);
            dEdx_arr[i] = jac_input_new[i] * output_std / input_std[i];
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
    NetworkForward_energy_grad<num>(
        tid, x_cur, output_dim, num_mid_layers, layer_arr, mWLst_cublas_dev,
        mbLst_cublas_dev, input_mean, input_std, output_mean, output_std,
        nullptr, nullptr, nullptr, &E, dEdx);
    energy_arr[tid] = E;
    for (int i = 0; i < num; i++)
    {
        dEdx_arr[tid][i] = dEdx[i];
    }
}

template <int num>
__global__ void NetworkForward_energy_grad_hess_global(
    int n_ele, devPtr<const const tCudaMatrix<float, num, 1>> x_arr,
    int output_dim, size_t num_mid_layers, devPtr<const uint> layer_arr_,
    devPtr<float *const> mWLst_cublas_dev,
    devPtr<float *const> mbLst_cublas_dev, devPtr<const float> input_mean,
    devPtr<const float> input_std, float output_mean, float output_std,
    // solution
    devPtr<float> energy_arr, devPtr<tCudaMatrix<float, num, 1>> dEdx_arr,
    devPtr<tCudaMatrix<float, num, num>> dE2dx2_arr)
{

    CUDA_function;

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // put layer into the shared mem
    constexpr int max_layer = 5;
    __shared__ unsigned int layer_arr_sm[max_layer];

    for (int i = threadIdx.x; i < num_mid_layers; i += blockDim.x)
        layer_arr_sm[i] = layer_arr_[i];
    __syncthreads();

    // put the net into shared mem
    __shared__ float network_param[3000];

    int network_param_w_st_pos[max_layer]; // max 10 layers
    int network_param_b_st_pos[max_layer]; // max 10 layers
    {
        int data_offset = 0;
        int total_layers = num_mid_layers + 1;
        int index = threadIdx.x;
        int stride = blockDim.x;
        for (size_t layer_id = 0; layer_id < total_layers; layer_id++)
        {
            int w_rows, w_cols, b_size;
            GetWbShape(layer_arr_sm, num_mid_layers, num, output_dim, layer_id,
                       w_rows, w_cols, b_size);
            float *w_data = mWLst_cublas_dev[layer_id];
            float *b_data = mbLst_cublas_dev[layer_id];

            // begin to copy per result
            network_param_w_st_pos[layer_id] = data_offset;
            for (int i = index; i < w_rows * w_cols; i += stride)
                network_param[data_offset + i] = w_data[i];
            data_offset += w_rows * w_cols;

            network_param_b_st_pos[layer_id] = data_offset;
            for (int i = index; i < b_size; i += stride)
                network_param[data_offset + i] = b_data[i];
            data_offset += b_size;
        }
    }
    __syncthreads();
    if (tid >= n_ele)
        return;

    float E = 0;
    float dEdx[num];
    tCudaMatrix<float, num, 1> x_cur = x_arr[tid];
    NetworkForward_energy_grad<num>(
        tid, x_cur, output_dim, num_mid_layers, layer_arr_sm, mWLst_cublas_dev,
        mbLst_cublas_dev, input_mean, input_std, output_mean, output_std,
        network_param, network_param_w_st_pos, network_param_b_st_pos, &E,
        dEdx);
    energy_arr[tid] = E;
    for (int i = 0; i < num; i++)
    {
        dEdx_arr[tid][i] = dEdx[i];
    }
    // calculate hessian
    float eps = 1e-3f;
#pragma unroll
    for (int i = 0; i < num; i++)
    {
        tCudaMatrix<float, num, 1> x_add = x_arr[tid];
        x_add[i] += eps;
        float dEdx_new[num];
        NetworkForward_energy_grad<num>(
            tid, x_add, output_dim, num_mid_layers, layer_arr_sm,
            mWLst_cublas_dev, mbLst_cublas_dev, input_mean, input_std,
            output_mean, output_std, network_param, network_param_w_st_pos,
            network_param_b_st_pos, &E, dEdx_new);
        for (int j = 0; j < num; j++)
        {
            // (j, i)
            dE2dx2_arr[tid](j, i) = (dEdx_new[j] - dEdx[j]) / eps;
            // printf("[kernel] hess(%d %d) = %.2f\n", i, j, dE2dx2_arr[tid](j,
            // i));
        }
    }
#pragma unroll
    for (int i = 0; i < num; i++)
#pragma unroll
        for (int j = i + 1; j < num; j++)
        {
            float mean = (dE2dx2_arr[tid](i, j) + dE2dx2_arr[tid](j, i)) / 2;

            dE2dx2_arr[tid](i, j) = mean;
            dE2dx2_arr[tid](j, i) = mean;
        }
}

template <int num>
__global__ void NetworkForward_energy_grad_hess_global_row_major(
    int n_ele, devPtr<const const tCudaMatrix<float, num, 1>> x_arr,
    int output_dim, size_t num_mid_layers, devPtr<const uint> layer_arr_,
    devPtr<float *const> mWLst_cublas_dev_rowmajor,
    devPtr<float *const> mbLst_cublas_dev, devPtr<const float> input_mean,
    devPtr<const float> input_std, float output_mean, float output_std,
    // solution
    devPtr<float> energy_arr, devPtr<tCudaMatrix<float, num, 1>> dEdx_arr,
    devPtr<tCudaMatrix<float, num, num>> dE2dx2_arr)
{

    CUDA_function;

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // put layer into the shared mem
    constexpr int max_layer = 5;
    __shared__ unsigned int layer_arr_sm[max_layer];

    for (int i = threadIdx.x; i < num_mid_layers; i += blockDim.x)
        layer_arr_sm[i] = layer_arr_[i];
    __syncthreads();

    // put the net into shared mem
    __shared__ float network_param[3000];

    int network_param_w_st_pos[max_layer]; // max 10 layers
    int network_param_b_st_pos[max_layer]; // max 10 layers
    {
        int data_offset = 0;
        int total_layers = num_mid_layers + 1;
        int index = threadIdx.x;
        int stride = blockDim.x;
        for (size_t layer_id = 0; layer_id < total_layers; layer_id++)
        {
            int w_rows, w_cols, b_size;
            GetWbShape(layer_arr_sm, num_mid_layers, num, output_dim, layer_id,
                       w_rows, w_cols, b_size);
            float *w_data = mWLst_cublas_dev_rowmajor[layer_id];
            float *b_data = mbLst_cublas_dev[layer_id];

            // begin to copy per result
            network_param_w_st_pos[layer_id] = data_offset;
            for (int i = index; i < w_rows * w_cols; i += stride)
                network_param[data_offset + i] = w_data[i];
            data_offset += w_rows * w_cols;

            network_param_b_st_pos[layer_id] = data_offset;
            for (int i = index; i < b_size; i += stride)
                network_param[data_offset + i] = b_data[i];
            data_offset += b_size;
        }
    }
    __syncthreads();
    if (tid >= n_ele)
        return;

    float E = 0;
    float dEdx[num];
    tCudaMatrix<float, num, 1> x_cur = x_arr[tid];
    // printf("[global] infer 1 begin\n");
    NetworkForward_energy_grad_row_major<num>(
        tid, x_cur, output_dim, num_mid_layers, layer_arr_sm, input_mean,
        input_std, output_mean, output_std, network_param,
        network_param_w_st_pos, network_param_b_st_pos, &E, dEdx);
    // printf("[global] infer 1 done\n");
    energy_arr[tid] = E;
    for (int i = 0; i < num; i++)
    {
        dEdx_arr[tid][i] = dEdx[i];
    }
    // calculate hessian
    float eps = 1e-3f;
#pragma unroll
    for (int i = 0; i < num; i++)
    {
        tCudaMatrix<float, num, 1> x_add = x_arr[tid];
        x_add[i] += eps;
        float dEdx_new[num];
        float tmp_E = 0;
        NetworkForward_energy_grad_row_major<num>(
            tid, x_add, output_dim, num_mid_layers, layer_arr_sm, input_mean,
            input_std, output_mean, output_std, network_param,
            network_param_w_st_pos, network_param_b_st_pos, &tmp_E, dEdx_new);
        for (int j = 0; j < num; j++)
        {
            // (j, i)
            dE2dx2_arr[tid](j, i) = (dEdx_new[j] - dEdx[j]) / eps;
            // printf("[kernel] hess(%d %d) = %.2f\n", i, j, dE2dx2_arr[tid](j,
            // i));
        }
    }
#pragma unroll
    for (int i = 0; i < num; i++)
#pragma unroll
        for (int j = i + 1; j < num; j++)
        {
            float mean = (dE2dx2_arr[tid](i, j) + dE2dx2_arr[tid](j, i)) / 2;

            dE2dx2_arr[tid](i, j) = mean;
            dE2dx2_arr[tid](j, i) = mean;
        }
}

#include "FullConnectedNetSingleScalarGPU.h"
void cFCNetworkSingleScalarGPU::forward_func_1d_energy_grad(
    const cCudaArray<tCudaVector1f> &x_arr, cCudaArray<float> &E_arr,
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

void cFCNetworkSingleScalarGPU::forward_func_2d_energy_grad(
    const cCudaArray<tCudaVector2f> &x_arr, cCudaArray<float> &E_arr,
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

void cFCNetworkSingleScalarGPU::forward_func_1d_energy_grad_hess(
    const cCudaArray<tCudaVector1f> &x_arr, cCudaArray<float> &E_arr,
    cCudaArray<tCudaVector1f> &dEdx_arr, cCudaArray<tCudaMatrix1f> &dE2dx2_arr)
{
    int N = x_arr.Size();
    // NetworkForward_energy_grad_shared_mem CUDA_at(N, 128)(
    NetworkForward_energy_grad_hess_global_row_major<1> CUDA_at(N, 512)(
        N, x_arr.Ptr(), 1, this->mLayersGPU.Size(), mLayersGPU.Ptr(),
        mWLst_row_major_dev.Ptr(), mbLst_cublas_dev.Ptr(), mInputMeanGPU.Ptr(),
        mInputStdGPU.Ptr(), mOutputMean, mOutputStd, E_arr.Ptr(),
        dEdx_arr.Ptr(), dE2dx2_arr.Ptr());
    CUDA_ERR("NetworkForward_energy_grad_hess_global_row_major 1");
}

#define MAX_NAME_LENGTH 64

typedef struct
{
    char name[MAX_NAME_LENGTH];
    cudaEvent_t start;
    cudaEvent_t stop;
} cuda_timer_t;
#define BEGIN_CUDA_TIMING(timer)                                               \
    cudaEventCreate(&(timer.start));                                           \
    cudaEventCreate(&(timer.stop));                                            \
    cudaEventRecord(timer.start, 0);

#define END_CUDA_TIMING(timer, name)                                           \
    cudaEventRecord(timer.stop, 0);                                            \
    cudaEventSynchronize(timer.stop);                                          \
    float msecTotal = 0.0f;                                                    \
    cudaEventElapsedTime(&msecTotal, timer.start, timer.stop);                 \
    printf("Time to execute %s: %f ms\n", name, msecTotal);                    \
    cudaEventDestroy(timer.start);                                             \
    cudaEventDestroy(timer.stop);

void cFCNetworkSingleScalarGPU::forward_func_2d_energy_grad_hess(
    const cCudaArray<tCudaVector2f> &x_arr, cCudaArray<float> &E_arr,
    cCudaArray<tCudaVector2f> &dEdx_arr, cCudaArray<tCudaMatrix2f> &dE2dx2_arr)
{

    int N = x_arr.Size();
    cuda_timer_t timer;
    BEGIN_CUDA_TIMING(timer);
    // NetworkForward_energy_grad_hess_global<2> CUDA_at(N, 256)(
    //     N, x_arr.Ptr(), 1, this->mLayersGPU.Size(), mLayersGPU.Ptr(),
    //     mWLst_cublas_dev.Ptr(), mbLst_cublas_dev.Ptr(), mInputMeanGPU.Ptr(),
    //     mInputStdGPU.Ptr(), mOutputMean, mOutputStd, E_arr.Ptr(),
    //     dEdx_arr.Ptr(), dE2dx2_arr.Ptr());

    printf("x cost %.3f MB\n", x_arr.Size() * 2.0 * 4 / 1e6);
    NetworkForward_energy_grad_hess_global_row_major<2> CUDA_at(N, 512)(
        N, x_arr.Ptr(), 1, this->mLayersGPU.Size(), mLayersGPU.Ptr(),
        mWLst_row_major_dev.Ptr(), mbLst_cublas_dev.Ptr(), mInputMeanGPU.Ptr(),
        mInputStdGPU.Ptr(), mOutputMean, mOutputStd, E_arr.Ptr(),
        dEdx_arr.Ptr(), dE2dx2_arr.Ptr());

    END_CUDA_TIMING(timer, "NetGPU");
    // CUDA_ERR("NetworkForward 2d");
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);

    // std::cout << "Registers per block: " << prop.regsPerBlock <<
    // std::endl; std::cout << "Registers per SM: " <<
    // prop.regsPerMultiprocessor
    //           << std::endl;

    // std::cout << "Shared memory per block: " << prop.sharedMemPerBlock
    //           << " bytes" << std::endl;

    // std::cout << "Number of multiprocessors: " <<
    // prop.multiProcessorCount
    //           << std::endl;

    // std::cout << "Shared memory per processor: "
    //           << prop.sharedMemPerMultiprocessor << " bytes" <<
    //           std::endl;

    // int numBlocks;       // Occupancy in terms of active blocks
    // int blockSize = 256; // The block size you want to launch your kernel
    // // with

    // // Get maximum number of active blocks per SM for your kernel
    // // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
    // kernelFunc,
    // //                                               blockSize, 0);

    // // std::cout << "Max blocks per processor: " << numBlocks <<
    // std::endl; exit(1);
}
// __global__ void NetworkForward_energy_grad_shared_mem(
//     int n_ele, devPtr<const tCudaVector1f> x_arr, int input_dim, int
//     output_dim, size_t num_mid_layers, devPtr<const uint> layer_arr,
//     devPtr<float *const> mWLst_cublas_dev_,
//     devPtr<float *const> mbLst_cublas_dev_, devPtr<const float>
//     input_mean, devPtr<const float> input_std, float output_mean, float
//     output_std,

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

//     // float *cur_multi_buf = cur_multi_buf_ + tid *
//     sizeof_cur_multi_buf;
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