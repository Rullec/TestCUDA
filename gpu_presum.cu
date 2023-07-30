#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaDevPtr.h"
#include <cfloat>
#include <climits>
__global__ void ApplyOffsetToBlocks(int num_of_data, devPtr<int> raw_data,
                                    devPtr<const int> per_block_offset_arr)
{
    CUDA_function;
    int tid_global = threadIdx.x + blockDim.x * blockIdx.x;
    int tid_local = threadIdx.x;
    if (tid_global >= num_of_data)
        return;
    raw_data[tid_global] += per_block_offset_arr[blockIdx.x];
}
__global__ void AddTwoArray(int num_of_data, devPtr<int> data0,
                            devPtr<const int> data1)
{
    CUDA_function;
    int tid_global = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid_global >= num_of_data)
        return;
    data0[tid_global] += data1[tid_global];
}

__global__ void CalcPresumGPUKernel_Blelloch(
    int num_of_data, devPtr<int> raw_data, devPtr<int> tar_data,
    devPtr<int> block_offset_final, bool output_to_block_offset_final,
    bool inplace_calculation, bool inclusive_prefix_sum)
{
    CUDA_function;

    int tid_global = threadIdx.x + blockDim.x * blockIdx.x;
    int tid_local = threadIdx.x;
    // if(tid_global == 0)
    //     printf("step1\n");
    extern __shared__ int smem[];
    smem[tid_local] = 0;

    int num_of_data_this_block = 0;
    if ((blockIdx.x + 1) * blockDim.x <= num_of_data)
        num_of_data_this_block = blockDim.x;
    else
    {
        num_of_data_this_block = num_of_data % blockDim.x;
    }

    int two_power_capacity = 1;
    while (two_power_capacity < num_of_data_this_block)
        two_power_capacity <<= 1;
    // printf(
    //     "block id %d block dim %d, num of data %d, num of data this block
    //     %d\n", blockIdx.x, blockDim.x, num_of_data, num_of_data_this_block);
    // printf("num of data %d num of data this block %d, block id %d, two power
    // cap %d\n", num_of_data, num_of_data_this_block, blockIdx.x,
    // two_power_capacity);
    if (tid_local >= two_power_capacity)
        return;

    if (tid_global < num_of_data)
        smem[tid_local] = raw_data[tid_global];
    __syncthreads();

    // if(tid_global == 0)
    //     printf("step1.3\n");
    // 1. up pass

    unsigned int stepsize = 1;
    while (stepsize < two_power_capacity)
    {
        unsigned int st = stepsize - 1;
        unsigned int gap = stepsize << 1;

        // if (tid == 0)
        //     printf("---forward stepsize %d gap %d---\n", stepsize, gap);
        if ((tid_local - st) % gap == 0 &&
            tid_local + stepsize < two_power_capacity)
        {

            smem[tid_local + stepsize] += smem[tid_local];
            // printf("forward smem[%d] += smem[%d], = %d\n", tid + stepsize,
            // tid,
            //        smem[tid + stepsize]);
        }
        stepsize <<= 1;
        __syncthreads();
    }

    // if(tid_global == 0)
    //     printf("step1.4\n");

    // 2. inverse pass
    int last_sum = smem[tid_local];
    // 2.1 set root to zero
    if (tid_local == (two_power_capacity - 1))
        smem[tid_local] = 0;

    // if(tid_global == 0)
    //     printf("step1.5\n");

    // now, stepsize == num_of_data
    while (stepsize >= 2)
    {
        unsigned int gap = stepsize;
        // if (tid == 0)
        //     printf("---inverse stepsize %d---\n", stepsize);
        if ((tid_local + 1) % stepsize == 0)
        {
            unsigned int child_id_left = tid_local - stepsize / 2;

            int left_val = smem[child_id_left];
            // printf("raw left[%d] = %d, right[%d] = %d\n", child_id_left,
            // smem[child_id_left], tid_local, smem[tid_local]);

            smem[child_id_left] = smem[tid_local];
            smem[tid_local] += left_val;
            // printf("after left[%d] = %d, right[%d] = %d\n", child_id_left,
            // smem[child_id_left], tid, smem[tid]);
        }
        stepsize >>= 1;
        __syncthreads();
    }

    // if (inclusive_prefix_sum == true)
    // {
    //     if (tid_global == 0)
    //         printf("inclusive is slow for this impl\n");
    //     if (tid_global < num_of_data)
    //         smem[tid_local] += raw_data[tid_global];
    // }

    // if(tid_global == 0)
    //     printf("step1.7\n");
    // 2. output
    // printf("[final] tid global %d, num of data %d\n", tid_global,
    // num_of_data);
    if (tid_global < num_of_data)
    {
        int cur_val = (inclusive_prefix_sum == false)
                          ? smem[tid_local]
                          : (((tid_local + 1) != num_of_data_this_block)
                                 ? (smem[tid_local + 1])
                                 : last_sum);

        if (inplace_calculation == false)
        {
            tar_data[tid_global] = cur_val;
            // printf("write to tar %d = %d\n", tid_global, smem[tid_local]);
        }
        else
        {
            raw_data[tid_global] = cur_val;
            // printf("write to raw %d = %d\n", tid_global, smem[tid_local]);
        }
    }

    if (output_to_block_offset_final == true)
    {
        if (tid_local == (num_of_data_this_block - 1))
        {
            block_offset_final[blockIdx.x] = last_sum;
        }
    }
    // if(tid_global == 0)
    //     printf("step2\n");
}

int calc_num_thread_for_scan_two_power(int num_of_data, int max_sm_size,
                                       int max_thread)
{

    int two_power_capacity = 1;
    while (two_power_capacity < num_of_data)
        two_power_capacity <<= 1;

    int num_thread = std::min(max_thread, max_sm_size / 4);
    num_thread = std::min(num_thread, two_power_capacity);
    num_thread = std::max(num_thread, 32);
    return num_thread;
}

void CalcPresumGPU(int shared_mem_size_bytes, int max_thread,
                   cCudaArray<int> &x_gpu, cCudaArray<int> &x_presum_gpu,
                   bool is_inclusive_prefix_sum)
{
    CUDA_function;
    int num_of_data = x_gpu.Size();

    int num_thread = calc_num_thread_for_scan_two_power(
        num_of_data, shared_mem_size_bytes, max_thread);

    int sm_bytes = num_thread * sizeof(int);
    printf("num thread %d\n", num_thread);
    if (num_thread < num_of_data)
    {
        // printf("error: current thread %d cannot handle data %d\n",
        // num_thread,
        //        num_of_data);
        // exit(1);

        cCudaArray<int> block_offset_buf;
        int num_blocks =
            (num_of_data / num_thread) + ((num_of_data % num_thread) != 0);
        block_offset_buf.Resize(num_blocks);
        printf("num of blocks %d, num o f thread %d, total num %d\n",
               num_blocks, num_thread, num_of_data);
        // 1. calculate for each block
        CalcPresumGPUKernel_Blelloch CUDA_at_SM(num_of_data, num_thread,
                                                sm_bytes)(
            num_of_data, x_gpu.Ptr(), x_presum_gpu.Ptr(),
            block_offset_buf.Ptr(), true, false, is_inclusive_prefix_sum);

        int num_of_thread_for_blocks = calc_num_thread_for_scan_two_power(
            num_blocks, shared_mem_size_bytes, max_thread);
        int sm_bytes_for_blocks = num_of_thread_for_blocks * sizeof(int);

        if (num_blocks > num_thread)
        {
            printf(
                "[error] the block offset arr size %d cannot be contained in a "
                "single block(max num thread %d)\b",
                num_blocks, num_thread);
            exit(1);
        }

        cCudaArray<int> del;
        CalcPresumGPUKernel_Blelloch CUDA_at_SM(
            num_blocks, num_of_thread_for_blocks, sm_bytes_for_blocks)(
            num_blocks, block_offset_buf.Ptr(), block_offset_buf.Ptr(),
            del.Ptr(), false, true, false);

        std::vector<int> block_offset_buf_cpu;
        block_offset_buf.Download(block_offset_buf_cpu);
        printf("block_offset_buf = ");
        for (auto &x : block_offset_buf_cpu)
            printf("%d ", x);
        printf("\n");
        CUDA_ERR("CalcPresumGPUKernel_Blelloch");
        ApplyOffsetToBlocks CUDA_at(num_of_data, num_thread)(
            num_of_data, x_presum_gpu.Ptr(), block_offset_buf.Ptr());
        CUDA_ERR("ApplyOffsetToBlocks");
    }
    else
    {
        // 1. determine the num of thread = sm_bytes
        cCudaArray<int> buf;
        CalcPresumGPUKernel_Blelloch CUDA_at_SM(num_of_data, num_thread,
                                                sm_bytes)(
            num_of_data, x_gpu.Ptr(), x_presum_gpu.Ptr(), buf.Ptr(), false,
            false, is_inclusive_prefix_sum);
        CUDA_ERR("CalcPresumGPU");
    }
}