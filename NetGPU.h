#pragma once
#include "Net.h"
#include "gpu_utils/CudaArray.h"
#include "utils/DefUtil.h"
class NetGPU : public Net
{
public:
    explicit NetGPU();
    virtual ~NetGPU();
    virtual void Init(std::string path);
    virtual void Init(const Json::Value &path);
    virtual void Init(int input_dim, int output_dim, int layers,
                      const std::vector<tMatrixX> &weight_lst,
                      const std::vector<tVectorX> &bias_lst, std::string act,
                      const tVectorX &input_mean, const _FLOAT &output_mean,
                      const tVectorX &input_std, const _FLOAT &output_std);

    // ============== input & output are all normed ===============
    virtual _FLOAT forward_normed(const tVectorX &x);
    virtual tVectorX calc_grad_wrt_input_normed(const tVectorX &x);
    virtual tMatrixX calc_hess_wrt_input_normed(const tVectorX &x);

    // ============== input & output are all unnormed ===============
    virtual _FLOAT forward_unnormed(const tVectorX &x);
    virtual tVectorX calc_grad_wrt_input_unnormed(const tVectorX &x);
    virtual tMatrixX calc_hess_wrt_input_unnormed(const tVectorX &x);

    // ============== batched inference ==============
    virtual std::vector<_FLOAT>
    forward_unnormed_batch(const tEigenArr<tVectorX> &x);

    virtual std::tuple<std::vector<float>, std::vector<tVectorXf>>
    forward_unnormed_energy_grad_batch(const tEigenArr<tVectorX> &x);

    virtual void forward_unnormed_energy_grad_hess_batch(
        const tEigenArr<tVectorX> &x, std::vector<float> &,
        std::vector<tVectorXf> &, std::vector<tMatrixXf> &);
    void AdjustGPUBuffer(int num_of_triangles);

    int mInputDim;
    int mOutputDim;

protected:
    void TransferDataToGPU();
    cCudaArray<unsigned int> mLayersGPU;
    cCudaArray<float *> mWLst_cublas_dev;
    cCudaArray<float *> mWLst_row_major_dev;
    cCudaArray<float *> mbLst_cublas_dev;
    cCudaArray<float> mInputMeanGPU, mInputStdGPU;

    // buffer size need to be adjusted
    // cCudaArray<float> mCompBufGPU_for_energy; //
    // const int TRIANGLE_COMP_BUF_SIZE = 100;
    cCudaArray<float> mTriangleEnergyGPU;  //
    cCudaArray<tCudaVector1f> mdEdxGPU_1d; //
    cCudaArray<tCudaVector2f> mdEdxGPU_2d; //

    cCudaArray<tCudaMatrix1f> mdE2dx2GPU_1d; //
    cCudaArray<tCudaMatrix2f> mdE2dx2GPU_2d; //

    // ================ buffer for grad ===============
    /*
        // for activation grad, max size = 100
        // for cur_multi, max_size = 2000
        // for two cur_result, max_size = 4000
    */
    cCudaArray<float> mCompBufGPU_for_grad[3];
    const int TRIANGLE_COMP_BUF_SIZE_FOR_GRAD[3] = {2000, 2000, 2000};

    // ================ energy inference ==============
    void forward_func_1d(const cCudaArray<tCudaVector1f> &x_arr,
                         cCudaArray<float> &E_arr);

    void forward_func_2d(const cCudaArray<tCudaVector2f> &x_arr,
                         cCudaArray<float> &E_arr);

    // ================ grad inference ================
    void forward_func_1d_energy_grad(const cCudaArray<tCudaVector1f> &x_arr,
                                     cCudaArray<float> &E_arr,
                                     cCudaArray<tCudaVector1f> &dEdx_arr);
    void forward_func_2d_energy_grad(const cCudaArray<tCudaVector2f> &x_arr,
                                     cCudaArray<float> &E_arr,
                                     cCudaArray<tCudaVector2f> &dEdx_arr);

    void
    forward_func_1d_energy_grad_hess(const cCudaArray<tCudaVector1f> &x_arr,
                                     cCudaArray<float> &E_arr,
                                     cCudaArray<tCudaVector1f> &dEdx_arr,
                                     cCudaArray<tCudaMatrix1f> &dE2dx2_arr);
    void
    forward_func_2d_energy_grad_hess(const cCudaArray<tCudaVector2f> &x_arr,
                                     cCudaArray<float> &E_arr,
                                     cCudaArray<tCudaVector2f> &dEdx_arr,
                                     cCudaArray<tCudaMatrix2f> &dE2dx2_arr);
};
SIM_DECLARE_PTR(NetGPU);