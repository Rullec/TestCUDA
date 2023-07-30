#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaUtil.h"
#include "utils/EigenUtil.h"

extern void CalcPresumGPU(int shared_mem_size_bytes, int max_thread,
                          cCudaArray<int> &x_gpu,
                          cCudaArray<int> &x_presum_gpu, bool is_inclusive_prefix_sum);
int main()
{
    int sm_bytes = cCudaUtil::GetSharedMemoryBytes();
    int max_thread = 32;
    bool is_inclusive_prefix_sum = false;
    int N = 1000;
    tVectorXi x = tVectorXi::Random(N);
    for (int i = 0; i < N; i++)
    {
        x[i] = x[i] % 10;
        // x[i] = 1;
    }
    tVectorXi x_presum_cpu = tVectorXi::Zero(N);

    if (is_inclusive_prefix_sum == false)
    {

        for (int i = 1; i < N; i++)
            x_presum_cpu[i] = x_presum_cpu[i - 1] + x[i - 1];
    }
    else
    {
        for (int i = 0; i < N; i++)
            x_presum_cpu[i] = ((i != 0) ? x_presum_cpu[i - 1] : 0) + x[i];
    }
    std::cout << "x = " << x.transpose() << std::endl;
    std::cout << "x cpu presum = " << x_presum_cpu.transpose() << std::endl;
    // 1. upload to GPU
    cCudaArray<int> x_gpu;
    cCudaArray<int> x_presum_gpu;
    x_gpu.Resize(N);
    x_gpu.Upload(x.data(), x.size());
    x_presum_gpu.Resize(N);
    // 2.
    CalcPresumGPU(sm_bytes, max_thread, x_gpu, x_presum_gpu, is_inclusive_prefix_sum);
    std::vector<int> x_presum_gpu_downloaded;
    x_presum_gpu.Download(x_presum_gpu_downloaded);
    printf("x gpu presum =");
    for (auto &x : x_presum_gpu_downloaded)
        printf("%d ", x);
    printf("\n");

    for (int i = 0; i < N; i++)
    {
        if (std::fabs(x_presum_cpu[i] - x_presum_gpu_downloaded[i]) > 1e-5)
        {
            printf("error: result cpu %d %d != %d\n", i, x_presum_cpu[i],
                   x_presum_gpu_downloaded[i]);
        }
    }
    printf("succ\n");
}