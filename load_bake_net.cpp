#include "FullConnectedNetSingleScalar.h"
#include "FullConnectedNetSingleScalarGPU.h"
#include "gpu_utils/CudaUtil.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include "utils/json/json.h"
#include <fstream>

void load_bake_info(int bake_samples, std::string file_info,
                    std::vector<tNetBakeInfoPtr> &info_1d_arr,
                    std::vector<tNetBakeInfoPtr> &info_2d_arr)
{
    // Open JSON file
    std::ifstream file_id(file_info);

    // Parse JSON file
    Json::Value root;
    file_id >> root;

    // Read the data
    info_1d_arr.clear();
    for (int i = 0; i < root["1DNet"].size(); i++)
    {
        printf("load 1D %d\n", i);

        tNetBakeInfoPtr info_1D = std::make_shared<tNetBakeInfo>();
        info_1D->samples = bake_samples;
        Json::Value cur_info = root["1DNet"][i];
        for (int j = 0; j < cur_info["x"].size(); j++)
        {
            info_1D->x_arr.push_back(tVectorXf::Ones(1) *
                                     cur_info["x"][j].asFloat());
            info_1D->e_arr.push_back(cur_info["e"][j].asFloat());
            info_1D->grad_arr.emplace_back(tVectorXf::Ones(1) *
                                           cur_info["grad"][j].asFloat());
            info_1D->hess_arr.push_back(tMatrixXf::Ones(1, 1) *
                                        cur_info["hess"][j].asFloat());
        }
        info_1d_arr.push_back(info_1D);
    }

    info_2d_arr.clear();
    for (int i = 0; i < root["2DNet"].size(); i++)
    {
        printf("load 2D %d\n", i);
        Json::Value cur_json = root["2DNet"][i];

        tNetBakeInfoPtr info_2D = std::make_shared<tNetBakeInfo>();
        info_2D->samples = bake_samples;
        int N = cur_json["x"].size();

        for (int j = 0; j < N; j++)
        {
            // printf("set x\n");
            info_2D->x_arr.emplace_back(tVector2f(
                cur_json["x"][j][0].asFloat(), cur_json["x"][j][1].asFloat()));
            // printf("set e\n");
            info_2D->e_arr.emplace_back(cur_json["e"][j].asFloat());

            // printf("set grad\n");
            info_2D->grad_arr.emplace_back(
                tVector2f(cur_json["grad"][j][0].asFloat(),
                          cur_json["grad"][j][1].asFloat()));

            // printf("set hess\n");
            tMatrix2f hess;
            hess(0, 0) = cur_json["hess"][j][0].asFloat();
            hess(1, 0) = cur_json["hess"][j][1].asFloat();
            hess(0, 1) = cur_json["hess"][j][2].asFloat();
            hess(1, 1) = cur_json["hess"][j][3].asFloat();
            info_2D->hess_arr.push_back(hess);
        }
        info_2d_arr.push_back(info_2D);
    }

    // Do the same for 2DNet
    file_id.close();
}
