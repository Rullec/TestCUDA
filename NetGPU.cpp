#include "NetGPU.h"
#include "gpu_utils/CublasUtil.h"
#include "utils/LogUtil.h"

NetGPU::NetGPU() {}
void NetGPU::Init(std::string path)
{
    Net::Init(path);
    TransferDataToGPU();
}
void NetGPU::Init(const Json::Value &path)
{
    Net::Init(path);
    TransferDataToGPU();
}
void NetGPU::Init(int input_dim, int output_dim, int layers,
                  const std::vector<tMatrixX> &weight_lst,
                  const std::vector<tVectorX> &bias_lst, std::string act,
                  const tVectorX &input_mean, const _FLOAT &output_mean,
                  const tVectorX &input_std, const _FLOAT &output_std)
{
    Net::Init(input_dim, output_dim, layers, weight_lst, bias_lst, act,
              input_mean, output_mean, input_std, output_std);
    TransferDataToGPU();
}

extern void VisitStats(const cCudaArray<float> &input_mean_gpu);

NetGPU::~NetGPU()
{
    // for (auto &x : mWLst_cublas_dev)
    // {
    //     cCublasUtil::ReleaseCudaMem(x);
    // }
    // for (auto &x : mbLst_cublas_dev)
    // {
    //     cCublasUtil::ReleaseCudaMem(x);
    // }
    // mWLst_cublas_dev.clear();
    // mbLst_cublas_dev.clear();
}
void NetGPU::TransferDataToGPU()
{
    // 1. layers
    std::vector<uint> layers(mLayers.begin(), mLayers.end());
    mLayersGPU.Upload(layers);

    // 2. w
    // int num_of_param_w = 0;
    std::vector<float *> w_cublas_ptr_dev(0), b_cublas_ptr_dev(0);
    for (auto &w : mWLst)
    {
        tMatrixXf w_float = w.cast<float>();
        // num_of_param_w += w.size();

        float *dev_ptr = nullptr;
        cublasStatus_t stat = cCublasUtil::eigenToCublas(w_float, &dev_ptr);
        SIM_ASSERT(stat == CUBLAS_STATUS_SUCCESS);
        w_cublas_ptr_dev.emplace_back(dev_ptr);
    }
    mWLst_cublas_dev.Upload(w_cublas_ptr_dev);

    w_cublas_ptr_dev.clear();
    for (auto &w : mWLst)
    {
        tMatrixXf new_w = w.transpose().cast<float>();
        float *dev_ptr = nullptr;
        cublasStatus_t stat = cCublasUtil::eigenToCublas(new_w, &dev_ptr);
        SIM_ASSERT(stat == CUBLAS_STATUS_SUCCESS);
        w_cublas_ptr_dev.emplace_back(dev_ptr);
    }
    mWLst_row_major_dev.Upload(w_cublas_ptr_dev);

    // 3. b
    // int num_of_param_b = 0;
    for (auto &b : mbLst)
    {
        float *dev_ptr = nullptr;
        tVectorXf b_float = b.cast<float>();
        // num_of_param_b += b.size();
        cublasStatus_t stat =
            cCublasUtil::eigenVectorToCublas(b_float, &dev_ptr);
        SIM_ASSERT(stat == CUBLAS_STATUS_SUCCESS);
        b_cublas_ptr_dev.emplace_back(dev_ptr);
    }
    mbLst_cublas_dev.Upload(b_cublas_ptr_dev);

    // printf("network total param %d, occupy %.1f kB",
    //        num_of_param_w + num_of_param_b,
    //        (num_of_param_w + num_of_param_b) * 4.0 / 1e3);
    std::cout << "[upload] input mean = " << mInputMean.transpose()
              << std::endl;
    std::cout << "[upload] input std = " << mInputStd.transpose() << std::endl;
    // 4. input mean & std
    std::vector<float> input_mean_cpu(mInputMean.begin(), mInputMean.end());
    // std::cout << input_mean_cpu[0] << std::endl;
    // exit(1);
    mInputMeanGPU.Upload(input_mean_cpu);

    // VisitStats(mInputMeanGPU);
    mInputStdGPU.Upload(std::vector<float>(mInputStd.begin(), mInputStd.end()));
    // VisitStats(mInputStdGPU);
}

_FLOAT NetGPU::forward_normed(const tVectorX &x) { return 0; }
tVectorX NetGPU::calc_grad_wrt_input_normed(const tVectorX &x)
{
    return tVectorX::Zero(0);
}
tMatrixX NetGPU::calc_hess_wrt_input_normed(const tVectorX &x)
{
    return tMatrixX::Zero(0, 0);
}

// ============== input & output are all unnormed ===============
_FLOAT NetGPU::forward_unnormed(const tVectorX &x) { return 0; }
tVectorX NetGPU::calc_grad_wrt_input_unnormed(const tVectorX &x)
{
    return tVectorX::Zero(0);
}
tMatrixX NetGPU::calc_hess_wrt_input_unnormed(const tVectorX &x)
{
    return tMatrixX::Zero(0, 0);
}

// extern std::vector<float>
// forward_func_1d(const cCudaArray<tCudaVector1f> &x_arr);

typedef Eigen::Matrix<float, 2, 1> tVector2f;
std::vector<_FLOAT>
NetGPU::forward_unnormed_batch(const tEigenArr<tVectorX> &x_arr)
{
    AdjustGPUBuffer(x_arr.size());

    int dim = x_arr[0].size();
    if (dim == 1)
    {
        std::vector<tCudaVector1f> x_arr_cpu(x_arr.size());
        std::transform(x_arr.begin(), x_arr.end(), x_arr_cpu.begin(),
                       [](const tVectorX &res)
                       {
                           tCudaVector1f ret;
                           ret[0] = res[0];
                           return ret;
                       });
        cCudaArray<tCudaVector1f> x_arr_gpu;
        x_arr_gpu.Upload(x_arr_cpu);

        // std::cout << "layers = " << mLayers.transpose() << std::endl;
        // printf("w lst shape = ");
        // for (auto &x : mWLst)
        // {
        //     printf("(%ld,%ld)\n", x.rows(), x.cols());
        //     std::cout << x << std::endl;
        // }
        // printf("\nb lst shape = ");
        // for (auto &x : mbLst)
        //     printf("(%ld), ", x.size());
        // printf("\n");

        forward_func_1d(x_arr_gpu, mTriangleEnergyGPU);
        std::vector<float> e_cpu;
        mTriangleEnergyGPU.Download(e_cpu);
        std::vector<_FLOAT> e_cpu_double(e_cpu.size());
        std::transform(e_cpu.begin(), e_cpu.end(), e_cpu_double.begin(),
                       [](float val) { return _FLOAT(val); });
        return e_cpu_double;
        // launch kernel to do calculation
    }
    else if (dim == 2)
    {
        std::vector<tCudaVector2f> x_arr_cpu(x_arr.size());
        std::transform(x_arr.begin(), x_arr.end(), x_arr_cpu.begin(),
                       [](const tVectorX &res)
                       {
                           tCudaVector2f ret;
                           ret[0] = res[0];
                           ret[1] = res[1];
                           return ret;
                       });
        cCudaArray<tCudaVector2f> x_arr_gpu;
        x_arr_gpu.Upload(x_arr_cpu);

        // std::cout << "layers = " << mLayers.transpose() << std::endl;
        // printf("w lst shape = ");
        // for (auto &x : mWLst)
        // {
        //     printf("(%ld,%ld)\n", x.rows(), x.cols());
        //     std::cout << x << std::endl;
        // }
        // printf("\nb lst shape = ");
        // for (auto &x : mbLst)
        //     printf("(%ld), ", x.size());
        // printf("\n");

        forward_func_2d(x_arr_gpu, mTriangleEnergyGPU);
        std::vector<float> e_cpu;
        mTriangleEnergyGPU.Download(e_cpu);
        std::vector<_FLOAT> e_cpu_double(e_cpu.size());
        std::transform(e_cpu.begin(), e_cpu.end(), e_cpu_double.begin(),
                       [](float val) { return _FLOAT(val); });
        return e_cpu_double;
    }
    else
    {
        SIM_ERROR("doesn't fit {}", dim);
        return {};
    }
}

void NetGPU::AdjustGPUBuffer(int num_of_triangles)
{
    bool need_update = mTriangleEnergyGPU.Size() < num_of_triangles;
    if (need_update)
    {
        mTriangleEnergyGPU.Resize(num_of_triangles);
        mdEdxGPU_1d.Resize(num_of_triangles);
        mdEdxGPU_2d.Resize(num_of_triangles);

        mdE2dx2GPU_1d.Resize(num_of_triangles);
        mdE2dx2GPU_2d.Resize(num_of_triangles);

        // mCompBufGPU_for_energy.Resize(TRIANGLE_COMP_BUF_SIZE *
        //                               num_of_triangles);

        // for (int i = 0; i < 3; i++)
        // {
        //     mCompBufGPU_for_grad[i].Resize(TRIANGLE_COMP_BUF_SIZE_FOR_GRAD[i]
        //     *
        //                                    num_of_triangles);
        // }
    }
}

std::tuple<std::vector<float>, std::vector<tVectorXf>>
NetGPU::forward_unnormed_energy_grad_batch(const tEigenArr<tVectorX> &x_arr)
{

    AdjustGPUBuffer(x_arr.size());

    int dim = x_arr[0].size();
    if (dim == 1)
    {
        std::vector<tCudaVector1f> x_arr_cpu(x_arr.size());
        std::transform(x_arr.begin(), x_arr.end(), x_arr_cpu.begin(),
                       [](const tVectorX &res)
                       {
                           tCudaVector1f ret;
                           ret[0] = res[0];
                           return ret;
                       });
        cCudaArray<tCudaVector1f> x_arr_gpu;
        x_arr_gpu.Upload(x_arr_cpu);

        // std::cout << "layers = " << mLayers.transpose() << std::endl;
        // printf("w lst shape = ");
        // for (auto &x : mWLst)
        // {
        //     printf("(%ld,%ld)\n", x.rows(), x.cols());
        //     std::cout << x << std::endl;
        // }
        // printf("\nb lst shape = ");
        // for (auto &x : mbLst)
        //     printf("(%ld), ", x.size());
        // printf("\n");

        forward_func_1d_energy_grad(x_arr_gpu, mTriangleEnergyGPU, mdEdxGPU_1d);
        std::vector<float> e_cpu;
        mTriangleEnergyGPU.Download(e_cpu);
        std::vector<tCudaVector1f> dedx_cpu;
        mdEdxGPU_1d.Download(dedx_cpu);
        std::vector<tVectorXf> dedx_eigen(dedx_cpu.size());

        // std::vector<_FLOAT> e_cpu_double(e_cpu.size());
        std::transform(dedx_cpu.begin(), dedx_cpu.end(), dedx_eigen.begin(),
                       [](const tCudaVector1f &val)
                       { return tVectorXf::Ones(1) * val[0]; });
        // return e_cpu_double;
        return std::make_tuple(e_cpu, dedx_eigen);
    }
    else
    {

        std::vector<tCudaVector2f> x_arr_cpu(x_arr.size());
        std::transform(x_arr.begin(), x_arr.end(), x_arr_cpu.begin(),
                       [](const tVectorX &res)
                       {
                           tCudaVector2f ret;
                           ret[0] = res[0];
                           ret[1] = res[1];
                           return ret;
                       });
        cCudaArray<tCudaVector2f> x_arr_gpu;
        x_arr_gpu.Upload(x_arr_cpu);

        // std::cout << "layers = " << mLayers.transpose() << std::endl;
        // printf("w lst shape = ");
        // for (auto &x : mWLst)
        // {
        //     printf("(%ld,%ld)\n", x.rows(), x.cols());
        //     std::cout << x << std::endl;
        // }
        // printf("\nb lst shape = ");
        // for (auto &x : mbLst)
        //     printf("(%ld), ", x.size());
        // printf("\n");

        forward_func_2d_energy_grad(x_arr_gpu, mTriangleEnergyGPU, mdEdxGPU_2d);
        std::vector<float> e_cpu;
        mTriangleEnergyGPU.Download(e_cpu);
        std::vector<tCudaVector2f> dedx_cpu;
        mdEdxGPU_2d.Download(dedx_cpu);
        std::vector<tVectorXf> dedx_eigen(dedx_cpu.size());

        // std::vector<_FLOAT> e_cpu_double(e_cpu.size());
        std::transform(dedx_cpu.begin(), dedx_cpu.end(), dedx_eigen.begin(),
                       [](const tCudaVector2f &val)
                       {
                           tVectorXf res = tVectorXf::Ones(2);
                           res[0] = val[0];
                           res[1] = val[1];
                           return res;
                       });
        // return e_cpu_double;
        return std::make_tuple(e_cpu, dedx_eigen);
    }
}
typedef Eigen::Matrix<float, 2, 2> tMatrix2f;
#include "utils/ProfUtil.h"
void NetGPU::forward_unnormed_energy_grad_hess_batch(
    const tEigenArr<tVectorX> &x_arr, std::vector<float> &E_arr,
    std::vector<tVectorXf> &grad_arr, std::vector<tMatrixXf> &hess_arr)
{
    cProfUtil::Begin("gpu/adjust_gpu_buf");
    AdjustGPUBuffer(x_arr.size());
    cProfUtil::End("gpu/adjust_gpu_buf");

    int dim = x_arr[0].size();
    if (dim == 1)
    {
        std::vector<tCudaVector1f> x_arr_cpu(x_arr.size());
        std::transform(x_arr.begin(), x_arr.end(), x_arr_cpu.begin(),
                       [](const tVectorX &res)
                       {
                           tCudaVector1f ret;
                           ret[0] = res[0];
                           return ret;
                       });
        cCudaArray<tCudaVector1f> x_arr_gpu;
        x_arr_gpu.Upload(x_arr_cpu);

        // std::cout << "layers = " << mLayers.transpose() << std::endl;
        // printf("w lst shape = ");
        // for (auto &x : mWLst)
        // {
        //     printf("(%ld,%ld)\n", x.rows(), x.cols());
        //     std::cout << x << std::endl;
        // }
        // printf("\nb lst shape = ");
        // for (auto &x : mbLst)
        //     printf("(%ld), ", x.size());
        // printf("\n");

        forward_func_1d_energy_grad_hess(x_arr_gpu, mTriangleEnergyGPU,
                                         mdEdxGPU_1d, mdE2dx2GPU_1d);
        mTriangleEnergyGPU.Download(E_arr);
        std::vector<tCudaVector1f> dedx_cpu;
        mdEdxGPU_1d.Download(dedx_cpu);

        std::vector<tCudaMatrix1f> de2dx2_cpu;
        mdE2dx2GPU_1d.Download(de2dx2_cpu);

        grad_arr.resize(dedx_cpu.size());
        hess_arr.resize(dedx_cpu.size());

        // std::vector<_FLOAT> E_arr_double(E_arr.size());
        std::transform(dedx_cpu.begin(), dedx_cpu.end(), grad_arr.begin(),
                       [](const tCudaVector1f &val)
                       { return tVectorXf::Ones(1) * val[0]; });

        std::transform(de2dx2_cpu.begin(), de2dx2_cpu.end(), hess_arr.begin(),
                       [](const tCudaMatrix1f &val)
                       {
                           tMatrixXf ret(1, 1);
                           ret(0, 0) = val(0, 0);
                           return ret;
                       });
        // return e_cpu_double;
        // return std::make_tuple(e_cpu, dedx_eigen, de2dx2_eigen);
    }
    else
    {
        cProfUtil::Begin("gpu/data_transform_and_upload");
        std::vector<tCudaVector2f> x_arr_cpu(x_arr.size());
        std::transform(x_arr.begin(), x_arr.end(), x_arr_cpu.begin(),
                       [](const tVectorX &res)
                       {
                           tCudaVector2f ret;
                           ret[0] = res[0];
                           ret[1] = res[1];
                           return ret;
                       });
        cCudaArray<tCudaVector2f> x_arr_gpu;
        x_arr_gpu.Upload(x_arr_cpu);
        cProfUtil::End("gpu/data_transform_and_upload");

        // std::cout << "layers = " << mLayers.transpose() << std::endl;
        // printf("w lst shape = ");
        // for (auto &x : mWLst)
        // {
        //     printf("(%ld,%ld)\n", x.rows(), x.cols());
        //     std::cout << x << std::endl;
        // }
        // printf("\nb lst shape = ");
        // for (auto &x : mbLst)
        //     printf("(%ld), ", x.size());
        // printf("\n");

        cProfUtil::Begin("gpu/net_infer");
        forward_func_2d_energy_grad_hess(x_arr_gpu, mTriangleEnergyGPU,
                                         mdEdxGPU_2d, mdE2dx2GPU_2d);
        cProfUtil::End("gpu/net_infer");
        cProfUtil::Begin("gpu/download");
        std::vector<tCudaVector2f> dedx_cpu;
        // std::vector<float> e_cpu;
        mTriangleEnergyGPU.Download(E_arr);
        mdEdxGPU_2d.Download(dedx_cpu);
        std::vector<tCudaMatrix2f> de2dx2_cpu;
        mdE2dx2GPU_2d.Download(de2dx2_cpu);

        cProfUtil::End("gpu/download");
        // grad_arr.resize(dedx_cpu.size());
        // hess_arr.resize(dedx_cpu.size());
        // for (int i = 0; i < x_arr.size(); i++)
        // {
        //     grad_arr[i].resize(2);
        //     hess_arr[i].resize(2, 2);
        // }
        cProfUtil::Begin("gpu/transform");

        OMP_PARALLEL_FOR(OMP_MAX_THREADS)
        for (int i = 0; i < x_arr.size(); i++)
        {
            const auto &tmp_hess = de2dx2_cpu[i];
            // grad_arr[i].resize(2);
            // hess_arr[i].resize(2, 2);
            //  << dedx_cpu[i][0], dedx_cpu[i][1];

            memcpy(grad_arr[i].data(), dedx_cpu[i].mData, sizeof(float) * 2);
            memcpy(hess_arr[i].data(), de2dx2_cpu[i].mData, sizeof(float) * 4);
            // .noalias() = tMatrix2f() tmp_hess(0, 0), tmp_hess(1, 0),
            // tmp_hess(0, 1),
            //     tmp_hess(1, 1);
        }
        // std::vector<_FLOAT> e_cpu_double(e_cpu.size());
        // std::transform(dedx_cpu.begin(), dedx_cpu.end(), grad_arr.begin(),
        //                [](const tCudaVector2f &val)
        //                {
        //                    tVectorXf res = tVectorXf::Ones(2);
        //                    res[0] = val[0];
        //                    res[1] = val[1];
        //                    return res;
        //                });

        // std::transform(de2dx2_cpu.begin(), de2dx2_cpu.end(),
        // hess_arr.begin(),
        //                [](const tCudaMatrix2f &val)
        //                {
        //                    tMatrixXf res = tMatrixXf::Ones(2, 2);
        //                    res(0, 0) = val(0, 0);
        //                    res(1, 0) = val(1, 0);
        //                    res(1, 1) = val(1, 1);
        //                    res(0, 1) = val(0, 1);
        //                    return res;
        //                });
        cProfUtil::End("gpu/transform");
        // return e_cpu_double;
        // return std::make_tuple(e_cpu, dedx_eigen, de2dx2_eigen);
    }
}