#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaUtil.h"
#include "utils/EigenUtil.h"
#include "utils/JsonUtil.h"
#include "utils/MathUtil.h"
#include "utils/json/json.h"
#include <type_traits>

template <typename dtype>
void CalcSegScan(int shared_mem_size_bytes, int max_thread,
                 cCudaArray<dtype> &x_gpu, cCudaArray<unsigned int> &seg_begin,
                 cCudaArray<dtype> &x_presum_gpu, bool is_inclusive);
typedef Eigen::VectorX<unsigned int> tVectorXui;

void test_seg_scan(int N, const tVectorXi &x, const tVectorXui &seg_begin_label,
                   int max_thread, bool is_inclusive, float seg_ratio)
{
    tVectorXi x_seg_scan_cpu(N);
    x_seg_scan_cpu.setZero();

    if (is_inclusive == false)
    {
        int cur_sum = x[0];
        for (int i = 1; i < N; i++)
        {
            if (seg_begin_label[i] == 1)
            {
                x_seg_scan_cpu[i] = 0;
                cur_sum = x[i];
            }
            else
            {
                x_seg_scan_cpu[i] = cur_sum;
                cur_sum += x[i];
            }
        }
    }
    else
    {
        int cur_sum = 0;
        for (int i = 0; i < N; i++)
        {
            if (seg_begin_label[i] == 1)
            {
                cur_sum = x[i];
            }
            else
            {
                cur_sum += x[i];
            }
            x_seg_scan_cpu[i] = cur_sum;
        }
    }

    // std::cout << "xj seg scan = " << x_seg_scan_cpu.transpose() << std::endl;
    cCudaArray<int> x_gpu;
    cCudaArray<uint> seg_begin_label_gpu;
    cCudaArray<int> x_gpu_seg_scan;
    x_gpu.Resize(N);
    seg_begin_label_gpu.Resize(N);
    x_gpu_seg_scan.Resize(N);

    x_gpu.Upload(x.data(), N);
    seg_begin_label_gpu.Upload(seg_begin_label.data(), N);

    CalcSegScan(cCudaUtil::GetSharedMemoryBytes(), max_thread, x_gpu,
                seg_begin_label_gpu, x_gpu_seg_scan, is_inclusive);

    // 2. download gpu result, print gpu result
    std::vector<int> x_gpu_seg_scan_result;
    x_gpu_seg_scan.Download(x_gpu_seg_scan_result);
    // std::cout << "x seg scan gpu = ";
    // for (int i = 0; i < x_gpu_seg_scan_result.size(); i++)
    // {
    //     auto x = x_gpu_seg_scan_result[i];
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;
    bool test_fail = false;
    for (int i = 0; i < x_gpu_seg_scan_result.size(); i++)
    {
        auto x = x_gpu_seg_scan_result[i];
        if (std::fabs(x - x_seg_scan_cpu[i]) > 1e-3)
        {
            printf("[error] cpu[%d] = %d, gpu = %d!\n", i, x_seg_scan_cpu[i],
                   x);
            test_fail = true;
        }
    }
    if (test_fail == true)
    {
        std::cout << std::endl;
        printf("[error] test failed for N %d, max_thread %d, is_inclusive %d, "
               "seg_ratio %.3f\n",
               N, max_thread, is_inclusive, seg_ratio);

        Json::Value root;
        root["x"] = cJsonUtil::BuildVectorJson<int>(x);
        root["seg"] = cJsonUtil::BuildVectorJson<unsigned int>(seg_begin_label);
        std::string filename = "error_case_seg_scan.json";
        cJsonUtil::WriteJson(filename, root);
        std::cout << "save to " << filename << std::endl;
        exit(1);
    }
    else
    {
        printf("[info] test succ for N %d, max_thread %d, is_inclusive %d, "
               "seg_ratio %.3f\n",
               N, max_thread, is_inclusive, seg_ratio);
    }
}
void test_seg_scan_wrapper(int N, int max_thread, bool is_inclusive,
                           float seg_ratio)
{
    tVectorXi x(N);
    x.setRandom();
    tVectorXui seg_begin_label(N);
    seg_begin_label.setZero();
    for (int i = 0; i < N; i++)
    {
        x[i] = std::fabs(x[i] % 10);
        // x[i] = 1;

        seg_begin_label[i] = cMathUtil::RandFloat(0, 1) < seg_ratio;
    }
    seg_begin_label[0] = 1;

    test_seg_scan(N, x, seg_begin_label, max_thread, is_inclusive, seg_ratio);
    // 1. print x, print seg, calculate cpu reuslt
    // std::cout << "x = " << x.transpose() << std::endl;
    // std::cout << "seg = " << seg_begin_label.transpose() << std::endl;
}

void test_seg_from_file(std::string json_file)
{
    Json::Value root;
    cJsonUtil::LoadJson(json_file, root);
    tVectorXi x = cJsonUtil::ReadVectorJson<int>(root["x"]);
    tVectorXui seg_begin_label =
        cJsonUtil::ReadVectorJson<unsigned int>(root["seg"]);
    std::cout << "seg_begin_label = " << seg_begin_label.transpose()
              << std::endl;
    test_seg_scan(x.size(), x, seg_begin_label, 32, true, 0.026);
}
int main()
{
    cMathUtil::SeedRand(0);
    // int max_thread = cCudaUtil::GetSharedMemoryBytes();
    int max_thread = 128;
    for (int N = 32; N < max_thread * max_thread; N += N / 10)
    {
        for (float ratio = 0.001; ratio < 0.5; ratio += 0.001)
        {
            test_seg_scan_wrapper(N, max_thread, true, ratio);
            test_seg_scan_wrapper(N, max_thread, false, ratio);
        }
    }

    // test_seg_from_file("error_34.json");
    // test_seg_scan(34, max_thread, true, 0.026);
}