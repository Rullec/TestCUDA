#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaUtil.h"
#include "utils/EigenUtil.h"
#include "utils/LogUtil.h"
#include "utils/MathUtil.h"
#include "utils/TimeUtil.hpp"

template <typename dtype>
dtype MinmaxReductionGPU(const cCudaArray<dtype> &data_arr, int ele_offset,
                         int ele_gap, int shared_mem_size_bytes, int max_thread,
                         cCudaArray<dtype> &comp_buf, bool is_max);

void TestMinMaxFloat()
{
    int gap = 3;
    int st = 2;
    cCudaArray<float> x_buf;
    x_buf.Resize(100);
    for (int test_num = 4; test_num < 1e6; test_num += 100)
    {
        tVectorXf x(test_num);
        x.setRandom();
        x *= 1000;
        // float max_x = x.minCoeff();
        float min_x = 1e9;
        for (int i = st; i < test_num; i += gap)
        {
            min_x = std::min(min_x, x[i]);
        }
        // printf("max x = %.6f\n", max_x);
        cCudaArray<float> x_gpu;
        x_gpu.Resize(test_num);
        x_gpu.Upload(x.data(), test_num);

        int sm_bytes = cCudaUtil::GetSharedMemoryBytes();
        int max_thread = cCudaUtil::GetMaxThreadPerBlock();

        cTimeUtil::Begin("gpu");
        float max_x_gpu = MinmaxReductionGPU<float>(x_gpu, st, gap, sm_bytes,
                                                    max_thread, x_buf, false);
        cTimeUtil::End("gpu");
        float diff = std::fabs(min_x - max_x_gpu);
        SIM_INFO("num {} gap {}, min x cpu {:.3f} gpu {:.3f}", test_num, gap,
                 min_x, max_x_gpu);
        if (diff > 1.0e-5)
        {
            SIM_ERROR("num {}, diff = {:.5f}", test_num, diff);
            exit(1);
        }
    }
}

typedef Eigen::Matrix<float, 3, 1> tVector3f;
void TestAABB()
{
    cCudaArray<float> x_buf;
    tVectorXf x_pos;
    cCudaArray<float> comp_buf;
    comp_buf.Resize(1000);
    for (int N = 10; N < 1e6; N += 100)
    {
        int sm_bytes = cCudaUtil::GetSharedMemoryBytes();
        int max_thread = cCudaUtil::GetMaxThreadPerBlock();
        x_pos.noalias() = tVectorXf::Random(3 * N);
        // calc aabb
        tVector3f aabb_min_cpu, aabb_max_cpu;
        aabb_min_cpu.setConstant(__FLT_MAX__);
        aabb_max_cpu.setConstant(-__FLT_MAX__);
        tVector3f aabb_min_gpu, aabb_max_gpu;

        cTimeUtil::Begin("cpu_aabb");
        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < 3; i++)
            {
                aabb_min_cpu[i] = std::min(aabb_min_cpu[i], x_pos[3 * j + i]);
                aabb_max_cpu[i] = std::max(aabb_max_cpu[i], x_pos[3 * j + i]);
            }
        }
        cTimeUtil::End("cpu_aabb");
        // x_pos.resize(3 * N, 1);
        x_buf.Resize(N * 3);
        x_buf.Upload(x_pos.data(), N * 3);

        cTimeUtil::Begin("gpu_aabb");
        for (int i = 0; i < 3; i++)
        {
            aabb_max_gpu[i] = MinmaxReductionGPU<float>(
                x_buf, i, 3, sm_bytes, max_thread, comp_buf, true);
            aabb_min_gpu[i] = MinmaxReductionGPU<float>(
                x_buf, i, 3, sm_bytes, max_thread, comp_buf, false);
        }
        cTimeUtil::End("gpu_aabb");

        float diff = (aabb_min_cpu - aabb_min_gpu).norm() +
                     (aabb_max_cpu - aabb_max_gpu).norm();
        if (diff > 1e-3)
        {
            SIM_ERROR("num {} aabb min cpu {} gpu {}; max cpu {} gpu {}", N,
                      aabb_min_cpu.transpose(), aabb_min_gpu.transpose(),
                      aabb_max_cpu.transpose(), aabb_max_gpu.transpose());
        }
        else
        {
            SIM_INFO("num {} succ aabb min {} max {}", N,
                     aabb_min_gpu.transpose(), aabb_max_gpu.transpose());
        }
    }
}
int main()
{
    cMathUtil::SeedRand(0);
    // TestMinMaxFloat();
    TestAABB();
}