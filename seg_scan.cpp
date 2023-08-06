#include "gpu_utils/CudaArray.h"
#include "gpu_utils/CudaUtil.h"
#include "utils/EigenUtil.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include "utils/MathUtil.h"
#include "utils/json/json.h"
#include <type_traits>

template <typename dtype>
void CalcSegScan(int shared_mem_size_bytes, int max_thread,
                 const cCudaArray<dtype> &x_gpu,
                 const cCudaArray<unsigned int> &seg_begin,
                 cCudaArray<dtype> &x_presum_gpu, bool is_inclusive);
typedef Eigen::VectorX<unsigned int> tVectorXui;

template <typename dtype>
void test_seg_scan(int N, const std::vector<dtype> &x,
                   const tVectorXui &seg_begin_label, int max_thread,
                   bool is_inclusive, float seg_ratio,
                   std::function<float(const dtype &, const dtype &)> get_norm)
{
    std::vector<dtype> x_seg_scan_cpu(N);

    if (is_inclusive == false)
    {
        dtype cur_sum = x[0];
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
        dtype cur_sum = 0;
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
    cCudaArray<dtype> x_gpu;
    cCudaArray<uint> seg_begin_label_gpu;
    cCudaArray<dtype> x_gpu_seg_scan;
    x_gpu.Resize(N);
    seg_begin_label_gpu.Resize(N);
    x_gpu_seg_scan.Resize(N);

    x_gpu.Upload(x.data(), N);
    seg_begin_label_gpu.Upload(seg_begin_label.data(), N);

    CalcSegScan(cCudaUtil::GetSharedMemoryBytes(), max_thread, x_gpu,
                seg_begin_label_gpu, x_gpu_seg_scan, is_inclusive);

    // 2. download gpu result, print gpu result
    std::vector<dtype> x_gpu_seg_scan_result;
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
        if (get_norm(x, x_seg_scan_cpu[i]) > 1e-3)
        {
            SIM_INFO("[error] cpu[{}] = {}, gpu = {}!", i, x_seg_scan_cpu[i],
                     x);
            test_fail = true;
        }
    }
    if (test_fail == true)
    {
        SIM_INFO(
            "[error] test failed for N {}, max_thread {}, is_inclusive {}, "
            "seg_ratio {:.3f}",
            N, max_thread, is_inclusive, seg_ratio);

        // Json::Value root;
        // auto result = convert_stdvec_to_eigenvec(x);
        // root["x"] =
        //     cJsonUtil::BuildVectorJson<dtype>();
        // root["seg"] = cJsonUtil::BuildVectorJson<unsigned
        // int>(seg_begin_label); std::string filename =
        // "error_case_seg_scan.json"; cJsonUtil::WriteJson(filename, root);
        // std::cout << "save to " << filename << std::endl;
        exit(1);
    }
    else
    {

        SIM_INFO("[info] test succ for N {}, max_thread {}, is_inclusive {}, "
                 "seg_ratio {:.3f}\n",
                 N, max_thread, is_inclusive, seg_ratio);
    }
}
template <typename dtype>
void test_seg_scan_wrapper(
    int N, int max_thread, bool is_inclusive, float seg_ratio,
    std::function<dtype()> get_random_value,
    std::function<float(const dtype &, const dtype &)> get_norm)
{
    std::vector<dtype> x(N);
    tVectorXui seg_begin_label(N);
    seg_begin_label.setZero();
    for (int i = 0; i < N; i++)
    {
        x[i] = get_random_value();
        seg_begin_label[i] = cMathUtil::RandFloat(0, 1) < seg_ratio;
    }
    seg_begin_label[0] = 1;

    test_seg_scan(N, x, seg_begin_label, max_thread, is_inclusive, seg_ratio,
                  get_norm);
    // 1. print x, print seg, calculate cpu reuslt
    // std::cout << "x = " << x.transpose() << std::endl;
    // std::cout << "seg = " << seg_begin_label.transpose() << std::endl;
}

template <typename dtype, int dim>
std::vector<dtype>
convert_eigenvec_to_stdvec(const Eigen::Matrix<dtype, dim, 1> &vec)
{
    std::vector<dtype> ret(vec.size());
    for (int i = 0; i < vec.size(); i++)
        ret[i] = vec[i];
    return ret;
}

template <typename dtype, int dims>
Eigen::Matrix<dtype, Eigen::Dynamic, 1> convert_stdvec_to_eigenvec(
    const tEigenArr<Eigen::Matrix<dtype, dims, 1>> &std_vec)
{
    Eigen::Matrix<dtype, Eigen::Dynamic, 1> x(dims * std_vec.size());
    for (int i = 0; i < std_vec.size(); i++)
    {
        x.segment(dims * i, dims) = std_vec[i];
    }

    return x;
}

template <typename dtype> void test_seg_from_file(std::string json_file)
{
    Json::Value root;
    cJsonUtil::LoadJson(json_file, root);
    std::vector<dtype> x =
        convert_eigenvec_to_stdvec(cJsonUtil::ReadVectorJson<dtype>(root["x"]));
    tVectorXui seg_begin_label =
        cJsonUtil::ReadVectorJson<unsigned int>(root["seg"]);
    std::cout << "seg_begin_label = " << seg_begin_label.transpose()
              << std::endl;
    test_seg_scan<dtype>(x.size(), x, seg_begin_label, 32, true, 0.026);
}

#include "gpu_utils/CudaArray.h"

void test_seg_scan_cuda2i(std::string json_path)
{
    int sm_bytes = cCudaUtil::GetSharedMemoryBytes();
    int max_thread = cCudaUtil::GetMaxThreadPerBlock();

    Json::Value root;
    cJsonUtil::LoadJson(json_path, root);
    tVectorXi input0 = cJsonUtil::ReadVectorJson<int>(root["vec2i_lst"]);
    tVectorXi input1 = cJsonUtil::ReadVectorJson<int>(root["bitmask_lst"]);
    SIM_INFO("{} {}", input0.size(), input1.size());
    std::vector<tCudaVector2i> input0_cpu = {};
    std::vector<unsigned int> input1_cpu = {};
    for (int i = 0; i < input0.size() / 2; i++)
    {
        tCudaVector2i tmp;
        tmp[0] = input0[2 * i];
        tmp[1] = input0[2 * i + 1];
        input0_cpu.push_back(tmp);

        input1_cpu.push_back(input1[i]);
    }

    for (int i = 0; i < 10; i++)
    {
        cCudaArray<tCudaVector2i> input0_gpu;
        cCudaArray<unsigned int> input1_gpu;
        input0_gpu.Upload(input0_cpu);
        input1_gpu.Upload(input1_cpu);

        cCudaArray<tCudaVector2i> output_gpu;
        output_gpu.Resize(input0_gpu.Size());
        CalcSegScan(sm_bytes, max_thread, input0_gpu, input1_gpu, output_gpu,
                    true);
        std::vector<tCudaVector2i> output_cpu;
        output_gpu.Download(output_cpu);
        std::cout << "39010 output =" << output_cpu[39010].transpose()
                  << std::endl;
    }
}

extern void TestRefAtomic();
int main()
{
    cMathUtil::SeedRand(0);
    // TestRefAtomic();
    // exit(1);
    // std::vector<std::string> path_lst = {
    //     "/home/xudong/NewDisk/Projects/test/test_python/0.json",
    //     "/home/xudong/NewDisk/Projects/test/test_python/1.json",
    //     "/home/xudong/NewDisk/Projects/test/test_python/2.json",
    //     "/home/xudong/NewDisk/Projects/test/test_python/3.json",
    //     "/home/xudong/NewDisk/Projects/test/test_python/4.json",
    //     "/home/xudong/NewDisk/Projects/test/test_python/5.json",
    //     "/home/xudong/NewDisk/Projects/test/test_python/6.json",
    //     "/home/xudong/NewDisk/Projects/test/test_python/7.json",
    //     "/home/xudong/NewDisk/Projects/test/test_python/8.json",
    //     "/home/xudong/NewDisk/Projects/test/test_python/9.json"};
    // for (auto path : path_lst)
    // {
    //     test_2i(path);
    // }
    // test_seg_scan_cuda2i();

    std::function<float()> get_random_value_float = []()
    { return cMathUtil::RandFloat(-1, 1); };

    std::function<float(const float &, const float &)> get_diff_float =
        [](const float &v0, const float &v1) { return std::fabs(v0 - v1); };

    std::function<tCudaVector2i()> get_random_value_2i = []()
    {
        tCudaVector2i res;
        res[0] = cMathUtil::RandInt(-10, 10);
        res[1] = cMathUtil::RandInt(-10, 10);
        return res;
    };

    std::function<float(const tCudaVector2i &, const tCudaVector2i &)>
        get_diff_2i = [](const tCudaVector2i &v0, const tCudaVector2i &v1)
    { return (v0 - v1).norm(); };

    // int max_thread = cCudaUtil::GetSharedMemoryBytes();
    int max_thread = 1024;
    int N = max_thread * max_thread;
    float ratio = 0.1;
    for (; N < max_thread * max_thread * max_thread; N += N / 10)
    {
        for (float ratio = 0.001; ratio < 0.5; ratio += 0.001)
        {
            test_seg_scan_wrapper<float>(N, max_thread, true, ratio,
                                         get_random_value_float,
                                         get_diff_float);
            test_seg_scan_wrapper<float>(N, max_thread, false, ratio,
                                         get_random_value_float,
                                         get_diff_float);
        }
    }
}