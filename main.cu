// #include "gpu_utils/CudaArray.h"
// #include "gpu_utils/CudaDef.h"
// #include "gpu_utils/CudaIntrinsic.h"
// #include <vector>

// __global__ void VisitStats(int N, devPtr<const int> input_mean_gpu)
// {
//     CUDA_function;
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < N)
//     {
//         printf("visit stats input mean [%d] = %d\n", tid,
//                input_mean_gpu[tid]);
//     }
// }

// void VisitStats(const cCudaArray<int> &input_mean_gpu)
// {
//     int N = input_mean_gpu.Size();
//     VisitStats CUDA_at(N, 128)(N, input_mean_gpu.Ptr());
//     CUDA_ERR("VisitStats");
// }
