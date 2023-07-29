#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaUtil.h"
#include "utils/EigenUtil.h"
#include "utils/LogUtil.h"
#include "utils/MathUtil.h"

template <typename dtype>
dtype MinmaxReductionGPU(const cCudaArray<dtype> &data_arr,
                         int shared_mem_size_bytes, int max_thread,
                         cCudaArray<dtype> &comp_buf, bool is_max);

int main()
{
    cMathUtil::SeedRand(0);
    for (int test_num = 1; test_num < 10000; test_num++)
    {
        tVectorXf x(test_num);
        x.setRandom();
        x *= 1000;
        float max_x = x.minCoeff();
        // printf("max x = %.6f\n", max_x);
        cCudaArray<float> x_gpu;
        x_gpu.Resize(test_num);
        x_gpu.Upload(x.data(), test_num);

        int sm_bytes = cCudaUtil::GetSharedMemoryBytes();
        int max_thread = cCudaUtil::GetMaxThreadPerBlock();
        cCudaArray<float> x_buf;

        float max_x_gpu =
            MinmaxReductionGPU<float>(x_gpu, sm_bytes, max_thread, x_buf, false);
        float diff = std::fabs(max_x - max_x_gpu);
        SIM_INFO("num {},max x cpu {:.3f} gpu {:.3f}", test_num, max_x,
                 max_x_gpu);
        if (diff > 1.0e-5)
        {
            SIM_ERROR("num {}, diff = {:.5f}", test_num, diff);
            exit(1);
        }
    }
}