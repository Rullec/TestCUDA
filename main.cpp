#include "FullConnectedNetSingleScalar.h"
#include "FullConnectedNetSingleScalarGPU.h"
extern std::vector<cFCNetworkSingleScalarPtr> mNet1dLstCPU, mNet2dLstCPU;
extern std::vector<cFCNetworkSingleScalarGPUPtr> mNet1dLstGPU, mNet2dLstGPU;
extern void BuildNet(std::string deep_model_json_path);

extern void Bake1DNet(cFCNetworkSingleScalarPtr net, int samples,
                      std::vector<float> &x_samples, std::vector<float> &e_arr,
                      std::vector<float> &grad_arr,
                      std::vector<float> &hess_arr);
extern void energy_grad_hess_verify();

extern std::string BakeNet(std::string raw_dnn_path, int sampels);

extern void load_bake_info(int samples, std::string file_info,
                           std::vector<tNetBakeInfoPtr> &info_1d_arr,
                           std::vector<tNetBakeInfoPtr> &info_2d_arr);
#include "utils/MathUtil.h"
tVectorX GenerateXInRange(const tMatrixX &x_range)
{
    tVectorX vec = tVectorX::Zero(x_range.cols());
    for (int i = 0; i < x_range.cols(); i++)
    {
        float coef = cMathUtil::RandFloat(0.1, 0.8);

        vec[i] = x_range(0, i) + coef * (x_range(1, i) - x_range(0, i));
    }
    return vec;
}

typedef Eigen::Matrix<float, 4, 1> tVector4f;
extern tVector4f CalcBilinearInterpolationCoef(float xmin, float xmax,
                                               float ymin, float ymax, float x,
                                               float y);

int main()
{
    // {
    //     float xmin = -1, xmax = 2;
    //     float ymin = 2, ymax = 3;
    //     CalcBilinearInterpolationCoef(xmin, xmax, ymin, ymax);
    // }
    // exit(1);
    std::string net_path = "dnn.json";
    BuildNet(net_path);
    energy_grad_hess_verify();
    return 0; 
    int bake_samples = 10;
    
    std::string bake_net_path = "baked_" + net_path;
    BakeNet(net_path, bake_samples);

    {
        std::vector<tNetBakeInfoPtr> info_1d_arr;
        std::vector<tNetBakeInfoPtr> info_2d_arr;
        load_bake_info(bake_samples, bake_net_path, info_1d_arr, info_2d_arr);

        // for (int i = 0; i < mNet1dLstCPU.size(); i++)
        // {
        //     mNet1dLstCPU[i]->SetBakedInfo(info_1d_arr[i]);
        //     mNet1dLstGPU[i]->SetBakedInfo(info_1d_arr[i]);

        //     // begin to do test inference
        //     auto &net = mNet1dLstCPU[i];

        //     tVectorX x = GenerateXInRange(net->mXRange);
        //     float final_E;
        //     tVectorXf final_grad;
        //     tMatrixXf final_hess;
        //     net->calc_E_grad_hess_unnormed_baked(x, final_E, final_grad,
        //                                          final_hess);
        // }
        for (int i = 0; i < mNet2dLstCPU.size(); i++)
        {
            mNet2dLstCPU[i]->SetBakedInfo(info_2d_arr[i]);
            mNet2dLstGPU[i]->SetBakedInfo(info_2d_arr[i]);

            // begin to do test inference
            auto &net = mNet2dLstCPU[i];

            tVectorX x = GenerateXInRange(net->mXRange);
            float final_E;
            tVectorXf final_grad;
            tMatrixXf final_hess;
            net->calc_E_grad_hess_unnormed_baked(x, final_E, final_grad,
                                                 final_hess);
        }
    }
}