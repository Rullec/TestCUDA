#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaDevPtr.h"
#include <cfloat>
#include <climits>

__global__ void AddTwoArray(devPtr<float> res) { atomicAdd(res, 1.0f); }
void TestRefAtomic()
{

    cCudaArray<float> res;
    res.Resize(1);
    AddTwoArray CUDA_at(1024, 512)(res.Ptr());
    std::vector<float> res_cpu;
    res.Download(res_cpu);
    printf("res = %.3f\n", res_cpu[0]);
}