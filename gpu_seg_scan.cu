#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaDevPtr.h"
#include <cfloat>
#include <climits>

template <typename dtype>
__global__ void CalcSegScan_Kernel(
    int num_of_data, devPtr<dtype> raw_data, devPtr<unsigned int> seg_begin_arr,
    devPtr<dtype> tar_data, devPtr<dtype> block_offset_final,
    devPtr<unsigned int> pos_of_first_1_in_this_block,
    bool output_to_block_offset_final, bool inplace_calculation,
    bool inclusive_prefix_sum, bool count_sum_of_flag,
    devPtr<unsigned int> sum_of_flag_per_block, bool output_pos_of_first_1)
{
    CUDA_function;

    int tid_global = threadIdx.x + blockDim.x * blockIdx.x;
    int tid_local = threadIdx.x;

    if (output_pos_of_first_1 == true && tid_local == 0)
    {
        pos_of_first_1_in_this_block[blockIdx.x] = 10000;
    }
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

    // if(tid_global == 0)
    //     printf("step1\n");
    // extern __shared__ dtype smem[];

    extern __shared__ __align__(sizeof(dtype)) unsigned char my_smem[];
    dtype *smem = reinterpret_cast<dtype *>(my_smem);
    dtype *smem_origin =
        reinterpret_cast<dtype *>(my_smem + sizeof(dtype) * two_power_capacity);

    bool *origin_flag = reinterpret_cast<bool *>(
        my_smem + 2 * sizeof(dtype) * two_power_capacity);
    bool *buf_flag = reinterpret_cast<bool *>(
        my_smem + 2 * sizeof(dtype) * two_power_capacity + two_power_capacity);

    smem[tid_local] = 0;
    smem_origin[tid_local] = 0;
    origin_flag[tid_local] = 0;
    buf_flag[tid_local] = 0;
    // printf(
    //     "block id %d block dim %d, num of data %d, num of data this block
    //     %d\n", blockIdx.x, blockDim.x, num_of_data, num_of_data_this_block);
    // printf("num of data %d num of data this block %d, block id %d, two power
    // cap %d\n", num_of_data, num_of_data_this_block, blockIdx.x,
    // two_power_capacity);
    if (tid_local >= two_power_capacity)
        return;

    if (tid_global < num_of_data)
    {
        smem[tid_local] = raw_data[tid_global];
        smem_origin[tid_local] = smem[tid_local];
        // printf("smem[%d] = %d\n", tid_local, smem[tid_local]);
        origin_flag[tid_local] = seg_begin_arr[tid_global];
        buf_flag[tid_local] = origin_flag[tid_local];
        // printf("flag[%d] = %d\n", tid_local, origin_flag[tid_local]);
    }
    __syncthreads();

    // if(tid_global == 0)
    //     printf("step1.3\n");
    // 1. up pass

    unsigned int stepsize = 1;
    while (stepsize < two_power_capacity)
    {
        unsigned int st = stepsize - 1;
        unsigned int gap = stepsize << 1;

        // if (tid_local == 0)
        //     printf("---forward stepsize %d gap %d---\n", stepsize, gap);
        if ((tid_local - st) % gap == 0 &&
            tid_local + stepsize < two_power_capacity)
        {

            if (buf_flag[tid_local + stepsize] == 0)
            {
                smem[tid_local + stepsize] += smem[tid_local];
                // printf(
                //     "[up,update val] buf[%d]==0, smem[%d] += smem[%d], =
                //     %d\n", tid_local + stepsize, tid_local + stepsize,
                //     tid_local, smem[tid_local + stepsize]);
            }
            buf_flag[tid_local + stepsize] |= buf_flag[tid_local];
            // printf("[up,update flag] buf[%d] |= buf[%d], = %d\n",
            //        tid_local + stepsize, tid_local,
            //        buf_flag[tid_local + stepsize]);
        }
        stepsize <<= 1;
        __syncthreads();
    }

    // if(tid_global == 0)
    //     printf("step1.4\n");

    // if (tid_local < num_of_data_this_block)
    //     printf("smem[%d] = %d\n", tid_local, smem[tid_local]);
    // 2. inverse pass
    dtype last_sum = smem[tid_local];
    // 2.1 set root to zero
    if (tid_local == (two_power_capacity - 1))
        smem[tid_local] = 0;

    // if(tid_global == 0)
    //     printf("step1.5\n");
    // now, stepsize == num_of_data
    while (stepsize >= 2)
    {
        // unsigned int gap = stepsize;
        // if (tid == 0)
        //     printf("---inverse stepsize %d---\n", stepsize);
        if ((tid_local + 1) % stepsize == 0)
        {
            unsigned int child_id_left = tid_local - stepsize / 2;

            dtype tmp = smem[child_id_left];

            // printf("raw left[%d] = %d, right[%d] = %d\n", child_id_left,
            // smem[child_id_left], tid_local, smem[tid_local]);
            smem[child_id_left] = smem[tid_local];
            if (origin_flag[child_id_left + 1] == 1)
            {
                smem[tid_local] = 0;
            }
            else if (buf_flag[child_id_left] != 0)
                smem[tid_local] = tmp;
            else
                smem[tid_local] += tmp;

            buf_flag[child_id_left] = 0;
            // smem[tid_local] += tmp * (seg_begin_arr[tid_local] != 1);
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
        dtype cur_val = (inclusive_prefix_sum)
                            ? (smem[tid_local] + smem_origin[tid_local])
                            : smem[tid_local];

        if (inplace_calculation == false)
        {
            tar_data[tid_global] = cur_val;

            // if constexpr (std::is_same_v<dtype, int> ||
            //               std::is_same_v<dtype, unsigned int>)
            //     printf("write to tar %d = %d\n", tid_global,
            //     smem[tid_local]);
            // else
            //     printf("write to tar %d = %.3f\n", tid_global,
            //     smem[tid_local]);
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
            block_offset_final[blockIdx.x] =
                smem[tid_local] + smem_origin[tid_local];
        }
    }

    if (count_sum_of_flag == true)
    {
        atomicAdd(&(sum_of_flag_per_block[blockIdx.x]), origin_flag[tid_local]);
    }

    if (output_pos_of_first_1 == true)
    {
        if (origin_flag[tid_local] == 1)
            atomicMin(&(pos_of_first_1_in_this_block[blockIdx.x]), tid_local);
    }
    // if(tid_global == 0)
    //     printf("step2\n");
}

int downcast_to_power_of_2(int val)
{
    if (val <= 1)
        return 1;

    // val >= 2
    int cnt = 0;
    while (val != 1)
    {
        cnt += 1;
        val >>= 1;
    }
    for (int i = 0; i < cnt; i++)
        val <<= 1;
    return val;
}
int calc_num_thread_for_seg_scan_two_power(int num_of_data, int max_sm_size,
                                           int max_thread)
{

    int two_power_capacity = 1;
    while (two_power_capacity < num_of_data)
        two_power_capacity <<= 1;

    /*
        what buffer we need?
        1. data buffer, two_power * 4, bytes
        2. flag buffer, two_power * 1, bytes
        3. flag origin buffer, two_power * 1, bytes

        total: two_power * 6 bytes
    */

    int num_thread = std::min(max_thread, max_sm_size / 10);
    num_thread = std::min(num_thread, two_power_capacity);
    num_thread = std::max(num_thread, 32);

    // if num_thread is not the power of 2, normalize it.
    num_thread = downcast_to_power_of_2(num_thread);

    return num_thread;
}

template <typename dtype>
__global__ void
ApplyOffsetToBlocksForSegScan(int num_of_data, devPtr<dtype> data_arr,
                              devPtr<const dtype> per_block_offset_arr,
                              devPtr<unsigned int> the_first_1_in_each_seg)
{
    CUDA_function;

    int tid_global = threadIdx.x + blockDim.x * blockIdx.x;
    int tid_local = threadIdx.x;

    if (tid_global >= num_of_data)
        return;
    if (tid_local < the_first_1_in_each_seg[blockIdx.x])
    {
        data_arr[tid_global] +=
            (blockIdx.x >= 1) ? (per_block_offset_arr[blockIdx.x - 1]) : 0;
    }
}

/*
    each value a thread

    1. get block id
    2. get offset
    3.
*/
// template <typename dtype>
// __global__ void ApplyOffsetToBlocks(int num_of_data, devPtr<dtype> raw_data,
//                                     devPtr<const dtype> per_block_offset_arr)
// {
//     CUDA_function;
//     int tid_global = threadIdx.x + blockDim.x * blockIdx.x;
//     // int tid_local = threadIdx.x;
//     if (tid_global >= num_of_data)
//         return;
//     raw_data[tid_global] += per_block_offset_arr[blockIdx.x];
// }

template <typename dtype>
void CalcSegScan(int shared_mem_size_bytes, int max_thread,
                 cCudaArray<dtype> &x_gpu, cCudaArray<unsigned int> &seg_begin,
                 cCudaArray<dtype> &x_presum_gpu, bool is_inclusive)
{
    int num_of_data = x_gpu.Size();

    int num_thread = calc_num_thread_for_seg_scan_two_power(
        num_of_data, shared_mem_size_bytes, max_thread);

    int sm_bytes = num_thread * (2 * sizeof(dtype) + 2);
    printf("total data num %d, thread num %d\n", num_of_data, num_thread);
    // std::cout << "num_thread = " << num_thread << std::endl;
    int num_blocks =
        (num_of_data / num_thread) + ((num_of_data % num_thread) != 0);
    if (num_blocks == 1)
    {
        // 1. determine the num of thread = sm_bytes
        cCudaArray<dtype> buf;

        cCudaArray<dtype> useless;

        cCudaArray<unsigned int> useless_ui;
        CalcSegScan_Kernel CUDA_at_SM(num_of_data, num_thread, sm_bytes)(
            num_of_data, x_gpu.Ptr(), seg_begin.Ptr(), x_presum_gpu.Ptr(),
            buf.Ptr(), useless_ui.Ptr(), false, false, is_inclusive, false,
            useless_ui.Ptr(), false);
        CUDA_ERR("CalcSegScan_Kernel");
    }
    else
    {
        cCudaArray<dtype> offset_of_each_block, offset_of_each_block_final;
        offset_of_each_block.Resize(num_blocks);
        offset_of_each_block_final.Resize(num_blocks);

        cCudaArray<unsigned int> sum_of_flag_each_block;
        cCudaArray<unsigned int> position_of_first_1_in_this_block;
        sum_of_flag_each_block.Resize(num_blocks);
        position_of_first_1_in_this_block.Resize(num_blocks);

        CalcSegScan_Kernel CUDA_at_SM(num_of_data, num_thread, sm_bytes)(
            num_of_data, x_gpu.Ptr(), seg_begin.Ptr(), x_presum_gpu.Ptr(),
            offset_of_each_block.Ptr(), position_of_first_1_in_this_block.Ptr(),
            true, false, is_inclusive, true, sum_of_flag_each_block.Ptr(),
            true);
        CUDA_ERR("CalcSegScan_Kernel");

        int num_thread_for_offset_calc = calc_num_thread_for_seg_scan_two_power(
            num_blocks, shared_mem_size_bytes, max_thread);

        int num_blocks_for_offset_calc =
            (num_blocks / num_thread_for_offset_calc) +
            ((num_blocks % num_thread_for_offset_calc) != 0);
        if (num_blocks_for_offset_calc > 1)
        {
            printf("[error] even the block offset %d cannot be contained in sm "
                   "%d\n",
                   num_blocks_for_offset_calc, num_thread_for_offset_calc);
            exit(1);
        }

        int sm_bytes_for_offset_calc =
            num_thread_for_offset_calc * (2 * sizeof(dtype) + 2);
        /*

                   bool output_to_block_offset_final
                   bool inplace_calculation
                   bool inclusive_prefix_sum
                   bool count_sum_of_flag
                   devPtr<unsigned int> sum_of_flag_per_block

        */
        // {
        //     std::vector<unsigned int> sum_of_flag_each_block_cpu;
        //     sum_of_flag_each_block.Download(sum_of_flag_each_block_cpu);
        //     std::cout << "sum_of_flag_each_block_cpu = ";
        //     for (auto &x : sum_of_flag_each_block_cpu)
        //         std::cout << x << " ";
        //     std::cout << std::endl;
        // }
        // {
        //     std::vector<dtype> offset_of_each_block_cpu;
        //     offset_of_each_block.Download(offset_of_each_block_cpu);
        //     std::cout << "offset_of_each_block_cpu = ";
        //     for (auto &x : offset_of_each_block_cpu)
        //         std::cout << x << " ";
        //     std::cout << std::endl;
        // }
        CalcSegScan_Kernel CUDA_at_SM(num_blocks, num_thread_for_offset_calc,
                                      sm_bytes_for_offset_calc)(
            num_blocks, offset_of_each_block.Ptr(),
            sum_of_flag_each_block.Ptr(), offset_of_each_block_final.Ptr(),
            offset_of_each_block.Ptr(), position_of_first_1_in_this_block.Ptr(),
            false, false, true, false, sum_of_flag_each_block.Ptr(), false);
        CUDA_ERR("CalcSegScan_Kernel for offset");
        // {
        //     std::vector<dtype> offset_of_each_block_final_cpu;
        //     offset_of_each_block_final.Download(offset_of_each_block_final_cpu);
        //     std::cout << "offset_of_each_block_final_cpu = ";
        //     for (auto &x : offset_of_each_block_final_cpu)
        //         std::cout << x << " ";
        //     std::cout << std::endl;
        // }
        ApplyOffsetToBlocksForSegScan<dtype> CUDA_at(num_of_data, num_thread)(
            num_of_data, x_presum_gpu.Ptr(), offset_of_each_block_final.Ptr(),
            position_of_first_1_in_this_block.Ptr());
        CUDA_ERR("ApplyOffsetToBlocksForSegScan for offset");
        // std::vector<dtype> offset_of_each_block_cpu;
        // cCudaArray<unsigned int> sum_of_flag_each_block_gpu;
        // sum_of_flag_each_block_gpu.Download(sum_of_flag_each_block);
        // offset_of_each_block.Download(offset_of_each_block_cpu);
    }
}

template void CalcSegScan(int shared_mem_size_bytes, int max_thread,
                          cCudaArray<int> &x_gpu,
                          cCudaArray<unsigned int> &seg_begin,
                          cCudaArray<int> &x_presum_gpu, bool is_inclusive);