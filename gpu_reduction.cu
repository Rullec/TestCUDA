#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaDevPtr.h"
#include <cfloat>
#include <climits>

// typedef int (*DeviceFuncPtr)(int, int);

template <typename dtype>
__global__ void ReductionKernel(int num_of_ele, int data_st, int data_gap,
                                devPtr<const dtype> data_arr,
                                devPtr<dtype> output, bool is_max)
{
    CUDA_function;
    // using func = is_max ? std::max : std::min;
    extern __shared__ dtype shared_mem[];

    int tid_global = threadIdx.x + blockIdx.x * blockDim.x;

    int tid_local = threadIdx.x;

    shared_mem[tid_local] = is_max ? (-FLT_MAX) : FLT_MAX;
    // 0. judge illegal
    if (tid_global >= num_of_ele)
        return;
    // 1. load the outer data into shared mem
    int block_id = blockIdx.x;
    // int num_of_thread_per_block = blockDim.x;

    shared_mem[tid_local] = data_arr[data_st + data_gap * tid_global];
    __syncthreads();

    // 2. begin to do reduction

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid_local < s)
        {
            shared_mem[tid_local] =
                is_max ? max(shared_mem[tid_local], shared_mem[tid_local + s])
                       : min(shared_mem[tid_local], shared_mem[tid_local + s]);
            // printf("s = %d, shared_mem[%d] = %.4f\n", s, tid_local,
            //        shared_mem[tid_local]);
        }
        __syncthreads();
    }

    // handle the last warp separately, since for it we can assume that all
    // threads are active
    if (tid_local < 32)
    {

        // when tid_local + 32 exceed the initialzied shared memory?
        volatile float *smem = shared_mem;
        // printf("0 smem[%d] = %.3f\n", tid_local, smem[tid_local]);
        smem[tid_local] = is_max ? (max(smem[tid_local], smem[tid_local + 32]))
                                 : min(smem[tid_local], smem[tid_local + 32]);
        // printf("1 smem[%d] = %.3f\n", tid_local, smem[tid_local]);
        smem[tid_local] = is_max ? (max(smem[tid_local], smem[tid_local + 16]))
                                 : min(smem[tid_local], smem[tid_local + 16]);
        // printf("2 smem[%d] = %.3f\n", tid_local, smem[tid_local]);
        smem[tid_local] = is_max ? (max(smem[tid_local], smem[tid_local + 8]))
                                 : min(smem[tid_local], smem[tid_local + 8]);
        // printf("3 smem[%d] = %.3f\n", tid_local, smem[tid_local]);
        smem[tid_local] = is_max ? (max(smem[tid_local], smem[tid_local + 4]))
                                 : min(smem[tid_local], smem[tid_local + 4]);
        smem[tid_local] = is_max ? (max(smem[tid_local], smem[tid_local + 2]))
                                 : min(smem[tid_local], smem[tid_local + 2]);
        smem[tid_local] = is_max ? (max(smem[tid_local], smem[tid_local + 1]))
                                 : min(smem[tid_local], smem[tid_local + 1]);
        // printf("final smem[%d] = %.3f\n", tid_local, smem[tid_local]);
    }

    // 3. write down
    if (tid_local == 0)
    {
        output[block_id] = shared_mem[0];
    }
}

template <typename dtype>
dtype MinmaxReductionGPU(const cCudaArray<dtype> &data_arr, int ele_st,
                         int ele_gap, int shared_mem_size_bytes, int max_thread,
                         cCudaArray<dtype> &comp_buf, bool is_max)
{
    int ele_bytes = sizeof(dtype);
    // printf("ele_bytes = %d\n", ele_bytes);
    int max_thread_per_block = shared_mem_size_bytes / ele_bytes;
    // printf("max_thread_per_block in hardware = %d\n", max_thread);
    // printf("max_thread_per_block in sm block = %d\n", max_thread_per_block);
    max_thread_per_block = std::min(max_thread, max_thread_per_block);
    // printf("max_thread_per_block = %d\n", max_thread_per_block);

    // 1. determine block_size
    unsigned int thread_per_block = 1;
    while (thread_per_block <= max_thread_per_block)
        thread_per_block <<= 1;
    thread_per_block >>= 1;

    // warp size > 32. warp size = 64.
    // we are rely on at least one single full warp to initialize sm[32] to -inf
    int num_of_ele = (data_arr.Size() - ele_st) / ele_gap +
                     (((data_arr.Size() - ele_st) % ele_gap) != 0);
    while ((thread_per_block >> 1) > num_of_ele &&
           (thread_per_block >> 1) >= 64)
        thread_per_block >>= 1;
    // printf("thread_per_block = %d\n", thread_per_block);

    int num_of_ele_cur = num_of_ele;
    bool is_first_iter = true;

    // printf("num_of_ele_cur = %d\n", num_of_ele_cur);

    int output_size = (num_of_ele_cur / thread_per_block) +
                      (num_of_ele_cur % thread_per_block);
    int total_buf_size = 2 * output_size;
    comp_buf.Resize(total_buf_size);

    int data_st_idx = 0;
    int buf_st_idx = output_size;

    while (true)
    {
        devPtr<const float> data_ptr = nullptr;
        if (is_first_iter)
        {
            data_ptr = data_arr.Ptr();
        }
        else
        {
            data_ptr = comp_buf.Ptr() + data_st_idx;
        };
        devPtr<float> buf_ptr = comp_buf.Ptr() + buf_st_idx;

        output_size = (num_of_ele_cur / thread_per_block) +
                      ((num_of_ele_cur % thread_per_block) != 0);
        int sm_bytes = thread_per_block * sizeof(dtype);
        // printf("output_size after =  %d\n", output_size);
        // printf("sm_bytes = %d\n", sm_bytes);
        ReductionKernel<dtype> CUDA_at_SM(num_of_ele_cur, thread_per_block,
                                          sm_bytes)(
            num_of_ele_cur, (is_first_iter ? ele_st : 0),
            (is_first_iter ? ele_gap : 1), data_ptr, buf_ptr, is_max);
        CUDA_ERR("ReductionKernel");
        num_of_ele_cur = output_size;
        is_first_iter = false;
        // printf("one iter done, num of ele cur = %d\n", num_of_ele_cur);
        if (num_of_ele_cur == 1)
        {
            // std::vector<float> x_cpu;
            // comp_buf.Download(x_cpu, 0);
            // return x_cpu[buf_st_idx];
            std::vector<float> x_cpu_new;
            comp_buf.Download(x_cpu_new, buf_st_idx, buf_st_idx + 1);
            return x_cpu_new[0];
        }

        // swap data and buf
        int tmp = data_st_idx;
        data_st_idx = buf_st_idx;
        buf_st_idx = tmp;
    }
}

template float MinmaxReductionGPU<float>(const cCudaArray<float> &data_arr,
                                         int shared_mem_size_bytes, int ele_st,
                                         int ele_gap, int max_thread,
                                         cCudaArray<float> &comp_buf,
                                         bool is_max);