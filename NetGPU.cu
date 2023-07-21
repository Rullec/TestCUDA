#include "NetGPU.cuh"
#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaDef.h"
#include "gpu_utils/CudaIntrinsic.h"
#include <vector>

__global__ void VisitStats(int N, devPtr<const float> arr_gpu)
{
    CUDA_function;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        printf("visit stats arr_gpu [%d] = %.3f\n", tid, arr_gpu[tid]);
    }
}

void VisitStats(const cCudaArray<float> &arr_gpu)
{
    int N = arr_gpu.Size();
    VisitStats CUDA_at(N, 128)(N, arr_gpu.Ptr());
    CUDA_ERR("VisitStats");
}

template <int num>
__global__ void NetworkForward(
    int n_ele, devPtr<const tCudaMatrix<float, num, 1>> x_arr, int input_dim,
    int output_dim, size_t num_mid_layers, devPtr<const uint> layer_arr,
    devPtr<float *const> mWLst_cublas_dev,
    devPtr<float *const> mbLst_cublas_dev, devPtr<const float> input_mean,
    devPtr<const float> input_std, float output_mean, float output_std,
    devPtr<float> comp_buf_energy, int num_of_buf_energy_per_T,
    devPtr<float> energy_arr)
{
    CUDA_function;

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_ele)
        return;

    devPtr<float> cur_triangle_st_comp_buf =
        comp_buf_energy + num_of_buf_energy_per_T * tid;
    int cur_comp_buf_offset = 0;

    using tVecType = tCudaMatrix<float, num, 1>;
    tVecType x_cur = x_arr[tid];
    Normalize<num>(x_cur, input_mean, input_std);

    // if (tid == 0)
    {
        // printf("-------------- CUDA kernel begin ------------\n");
        // mInputMeanGPU[0];
        // printf("t %ld, x(after normalized) = %.3f\n", tid, x_cur[0]);
        // 1. show input mean, std

        // // printf("input mean %.3f std %.3f\n", mInputMeanGPU[0],
        // mInputStdGPU[0]);
        // printf("num layers = %ld, layer = ", num_mid_layers);
        // for (size_t i = 0; i < num_mid_layers; i++)
        // printf(" %d", layer_arr[i]);
        // printf("\n");
        int total_layers = num_mid_layers + 1;
        // printf("w.b shape = \n");
        int prev_rows = 0;
        for (size_t layer_id = 0; layer_id < total_layers; layer_id++)
        {
            int w_rows, w_cols, b_size;
            GetWbShape(layer_arr, num_mid_layers, input_dim, output_dim,
                       layer_id, w_rows, w_cols, b_size);
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
            if (layer_id == 0)
            {
                // printf("layer %ld z = ", layer_id);
                // PrintCublasVec(w_rows, cur_triangle_st_comp_buf);

                ApplySoftplus(cur_triangle_st_comp_buf, w_rows);
                // printf("layer %ld x next = ", layer_id);
                // PrintCublasVec(w_rows, cur_triangle_st_comp_buf);
            }
            else if (layer_id != total_layers - 1)
            {
                ApplySoftplus(cur_triangle_st_comp_buf + cur_comp_buf_offset,
                              w_rows);
                // printf("layer %ld x cur = ", layer_id);
                // PrintCublasVec(w_cols, cur_triangle_st_comp_buf +
                //                            cur_comp_buf_offset - prev_rows);

                // printf("layer %ld z = ", layer_id);
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

        energy_arr[tid] =
            (cur_triangle_st_comp_buf + cur_comp_buf_offset - prev_rows)[0] *
                output_std +
            output_mean;
        // printf("energy %ld = %.3f\n", tid, energy_arr[tid]);
    }
}

#include "NetGPU.h"
void NetGPU::forward_func_1d(const cCudaArray<tCudaVector1f> &x_arr,
                             cCudaArray<float> &E_arr)
{
    int N = x_arr.Size();
    NetworkForward CUDA_at(N, 128)(
        N, x_arr.Ptr(), 1, 1, this->mLayersGPU.Size(), mLayersGPU.Ptr(),
        mWLst_cublas_dev.Ptr(), mbLst_cublas_dev.Ptr(), mInputMeanGPU.Ptr(),
        mInputStdGPU.Ptr(), mOutputMean, mOutputStd,
        mCompBufGPU_for_energy.Ptr(), TRIANGLE_COMP_BUF_SIZE, E_arr.Ptr());
    CUDA_ERR("NetworkForward 1d");
}
void NetGPU::forward_func_2d(const cCudaArray<tCudaVector2f> &x_arr,
                             cCudaArray<float> &E_arr)
{
    int N = x_arr.Size();
    NetworkForward CUDA_at(N, 128)(
        N, x_arr.Ptr(), 2, 1, this->mLayersGPU.Size(), mLayersGPU.Ptr(),
        mWLst_cublas_dev.Ptr(), mbLst_cublas_dev.Ptr(), mInputMeanGPU.Ptr(),
        mInputStdGPU.Ptr(), mOutputMean, mOutputStd,
        mCompBufGPU_for_energy.Ptr(), TRIANGLE_COMP_BUF_SIZE, E_arr.Ptr());
    CUDA_ERR("NetworkForward 2d");
}
