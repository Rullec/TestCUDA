#include "gpu_utils/CudaUtil.h"
#include <cuda_runtime_api.h>
#include <assert.h>

// __global__ void getRegistersPerBlock()
// {
//     // Use the __regCount__ intrinsic to get the maximum registers per thread
//     int registersPerThread = __regCount__;

//     // Use the __launch_bounds__ intrinsic to get the maximum threads per
//     block int maxThreadsPerBlock = __launch_bounds__.x;

//     // Calculate the size of registers per block
//     int registersPerBlock = registersPerThread * maxThreadsPerBlock;

//     // Print the result from the first thread in the block
//     if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
//     {
//         printf("Registers per block: %d\n", registersPerBlock);
//     }
// }

size_t cCudaUtil::GetSharedMemoryBytes(int device_id)
{
    cudaDeviceProp prop;
    int count;

    cudaGetDeviceCount(&count);

    assert(device_id < count);
    cudaGetDeviceProperties(&prop, device_id);
    size_t sm_size = prop.sharedMemPerBlock;
    return sm_size;
}
// static size_t cCudaUtil::GetRegisterBytes();