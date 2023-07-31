#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaUtil.h"
#include "utils/EigenUtil.h"
#include <type_traits>

template <typename dtype>
void CalcPresumGPU(int shared_mem_size_bytes, int max_thread,
                   cCudaArray<dtype> &x_gpu, cCudaArray<dtype> &x_presum_gpu,
                   bool is_inclusive_prefix_sum);
template <typename dtype> void Test(int N)
{
    int sm_bytes = cCudaUtil::GetSharedMemoryBytes();
    int max_thread = 32;
    bool is_inclusive_prefix_sum = false;
    using vec = Eigen::Matrix<dtype, -1, 1>;
    vec x = vec::Random(N);
    if constexpr (std::is_same_v<dtype, int> ||
                  std::is_same_v<dtype, unsigned int>)
    {

        for (int i = 0; i < N; i++)
            x[i] = x[i] % 10;
    }

    vec x_presum_cpu = vec::Zero(N);

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
    cCudaArray<dtype> x_gpu;
    cCudaArray<dtype> x_presum_gpu;
    x_gpu.Resize(N);
    x_gpu.Upload(x.data(), x.size());
    x_presum_gpu.Resize(N);
    // 2.
    CalcPresumGPU<dtype>(sm_bytes, max_thread, x_gpu, x_presum_gpu,
                         is_inclusive_prefix_sum);
    std::vector<dtype> x_presum_gpu_downloaded;
    x_presum_gpu.Download(x_presum_gpu_downloaded);
    printf("x gpu presum =");
    for (auto &x : x_presum_gpu_downloaded)
    {

        if constexpr (std::is_same_v<dtype, int> ||
                      std::is_same_v<dtype, unsigned int>)
            printf("%d ", x);
        else
            printf("%.5f ", x);
    }
    printf("\n");

    bool test_failed = false;
    for (int i = 0; i < N; i++)
    {
        if (std::fabs(x_presum_cpu[i] - x_presum_gpu_downloaded[i]) > 1e-3)
        {
            test_failed = true;

            if constexpr (std::is_same_v<dtype, int> ||
                          std::is_same_v<dtype, unsigned int>)
            {
                printf("error: result cpu %d %d != %d\n", i, x_presum_cpu[i],
                       x_presum_gpu_downloaded[i]);
            }
            else
            {
                printf("error: result cpu %d %.5f != %.5f\n", i,
                       x_presum_cpu[i], x_presum_gpu_downloaded[i]);
            }
        }
    }
    if (test_failed == false)
        printf("succ\n");
    else
    {
        printf("failed!\n");
        exit(1);
    }
}
int main()
{
    // Test<int>(100);
    Test<float>(1000);
}