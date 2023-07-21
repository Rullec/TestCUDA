#include "Net.h"
#include "NetGPU.h"
#include "utils/FileUtil.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include "utils/json/json.h"

template <typename dtype>
int GetOffsetFromIdx(const std::vector<dtype> &arr, const dtype &val)
{
    return std::find(arr.begin(), arr.end(), val) - arr.begin();
}

std::vector<NetPtr> mNet1dLstCPU, mNet2dLstCPU;
std::vector<NetGPUPtr> mNet1dLstGPU, mNet2dLstGPU;
void BuildNet(std::string deep_model_json_path)
{
    // load
    if (false == cFileUtil::ExistsFile(deep_model_json_path))
    {
        SIM_ERROR("model path {} does not exist", deep_model_json_path);
    }
    std::vector<int> index_1d_lst = {0, 1, 2, 3, 5};
    std::vector<tVector2i> index_2d_lst = {
        {0, 1}, {0, 2}, {0, 3}, {0, 5}, {1, 2}, {1, 3}, {1, 5}, {2, 3}, {2, 5}};
    Json::Value root;
    SIM_ASSERT(cJsonUtil::LoadJson(deep_model_json_path, root));

    Json::Value json_1d_lst = cJsonUtil::ParseAsValue("1d", root);
    Json::Value json_2d_lst = cJsonUtil::ParseAsValue("2d", root);

    mNet1dLstCPU.resize(index_1d_lst.size(), nullptr);
    mNet2dLstCPU.resize(index_2d_lst.size(), nullptr);

    mNet1dLstGPU.resize(index_1d_lst.size(), nullptr);
    mNet2dLstGPU.resize(index_2d_lst.size(), nullptr);
    for (auto &x : json_1d_lst)
    {
        auto path = cJsonUtil::ParseAsString("raw_pth_model_path", x);
        int idx_1d = cJsonUtil::ParseAsInt("id", x);
        Json::Value net_val = cJsonUtil::ParseAsValue("net", x);
        int offset_1d = GetOffsetFromIdx(index_1d_lst, idx_1d);
        SIM_ASSERT(mNet1dLstCPU[offset_1d] == nullptr);

        Json::Value comment_json;
        comment_json["idx_1d"] = idx_1d;
        comment_json["offset_1d"] = offset_1d;
        comment_json["model_path"] = path;

        mNet1dLstCPU[offset_1d] = std::make_shared<Net>();
        mNet1dLstCPU[offset_1d]->Init(net_val);
        mNet1dLstCPU[offset_1d]->SetComment(
            cJsonUtil::WriteJson2Str(comment_json));

        mNet1dLstGPU[offset_1d] = std::make_shared<NetGPU>();
        mNet1dLstGPU[offset_1d]->Init(net_val);
        mNet1dLstGPU[offset_1d]->SetComment(
            cJsonUtil::WriteJson2Str(comment_json));

        printf("[info] Load DNN dof %d from %s\n", idx_1d,
               cFileUtil::GetFilename(path).c_str());
    }

    for (auto &x : json_2d_lst)
    {
        auto path = cJsonUtil::ParseAsString("raw_pth_model_path", x);
        tVector2i idx_2d = cJsonUtil::ReadVectorJson<int>("id", x, 2);
        Json::Value net_val = cJsonUtil::ParseAsValue("net", x);
        int offset_2d = GetOffsetFromIdx(index_2d_lst, idx_2d);
        SIM_ASSERT(mNet2dLstCPU[offset_2d] == nullptr);

        Json::Value comment_json;
        comment_json["idx_2d"] = Json::arrayValue;
        comment_json["idx_2d"].append(idx_2d[0]);
        comment_json["idx_2d"].append(idx_2d[1]);

        comment_json["offset_2d"] = offset_2d;
        comment_json["model_path"] = path;

        mNet2dLstCPU[offset_2d] = std::make_shared<Net>();
        mNet2dLstCPU[offset_2d]->Init(net_val);
        mNet2dLstCPU[offset_2d]->SetComment(
            cJsonUtil::WriteJson2Str(comment_json));

        mNet2dLstGPU[offset_2d] = std::make_shared<NetGPU>();
        mNet2dLstGPU[offset_2d]->Init(net_val);
        mNet2dLstGPU[offset_2d]->SetComment(
            cJsonUtil::WriteJson2Str(comment_json));
        printf("[info] Load DNN dof %d %d from %s\n", idx_2d[0], idx_2d[1],
               cFileUtil::GetFilename(path).c_str());
    }
}

extern void VisitStats(const cCudaArray<int> &input_mean_gpu);
#include "utils/TimeUtil.hpp"
void energy_verify()
{

    // energy debug
    if (true)
    {
        // test 1d
        int test_n = 5;
        auto net_cpu = mNet1dLstCPU[0];
        tEigenArr<tVectorX> batch_x = {};
        std::vector<_FLOAT> E_cpu_arr = {};
        cTimeUtil::Begin("cpu");
        for (int i = 0; i < test_n; i++)
        {
            tVectorX x = tVectorX::Random(1);
            batch_x.emplace_back(x);
            E_cpu_arr.push_back(net_cpu->forward_unnormed(x));
        }
        _FLOAT cpu_elasped = cTimeUtil::End("cpu");

        auto net_gpu = mNet1dLstGPU[0];

        std::vector<_FLOAT> E_gpu = net_gpu->forward_unnormed_batch(batch_x);

        cTimeUtil::Begin("gpu");
        E_gpu = net_gpu->forward_unnormed_batch(batch_x);
        _FLOAT gpu_elasped = cTimeUtil::End("gpu");
        // _FLOAT E_gpu = net_gpu->forward_unnormed(x);

        for (int i = 0; i < test_n; i++)
            printf("[%d] E cpu %.2f gpu %.2f\n", i, E_cpu_arr[i], E_gpu[i]);
        printf("[1d] cpu %.2f gpu %.2f\n", cpu_elasped, gpu_elasped);
    }
    else
    {
        // test 2d
        int test_n = 10000;
        auto net_cpu = mNet2dLstCPU[0];
        tEigenArr<tVectorX> batch_x = {};
        std::vector<_FLOAT> E_cpu_arr = {};
        cTimeUtil::Begin("cpu");
        for (int i = 0; i < test_n; i++)
        {
            tVectorX x = tVectorX::Random(2);
            batch_x.emplace_back(x);
            E_cpu_arr.push_back(net_cpu->forward_unnormed(x));
        }
        _FLOAT cpu_elasped = cTimeUtil::End("cpu");

        auto net_gpu = mNet2dLstGPU[0];
        std::vector<_FLOAT> E_gpu = net_gpu->forward_unnormed_batch(batch_x);

        cTimeUtil::Begin("gpu");
        E_gpu = net_gpu->forward_unnormed_batch(batch_x);
        _FLOAT gpu_elasped = cTimeUtil::End("gpu");
        // _FLOAT E_gpu = net_gpu->forward_unnormed(x);

        for (int i = 0; i < test_n; i++)
            printf("[%d] E cpu %.2f gpu %.2f\n", i, E_cpu_arr[i], E_gpu[i]);
        printf("[2d] cpu %.2f gpu %.2f\n", cpu_elasped, gpu_elasped);
    }
}

void energy_grad_verify()
{
    // test 1d
    int test_n = 50000;
    auto net_cpu = mNet1dLstCPU[0];
    tEigenArr<tVectorX> batch_x = {};
    std::vector<float> E_cpu_arr = {};
    std::vector<tVectorX> dEdx_cpu_arr = {};
    cTimeUtil::Begin("cpu");
    for (int i = 0; i < test_n; i++)
    {
        tVectorX x = tVectorX::Random(1);
        batch_x.emplace_back(x);
        float E = net_cpu->forward_unnormed(x);
        tVectorX dEdx = net_cpu->calc_grad_wrt_input_unnormed(x);

        E_cpu_arr.emplace_back(E);
        dEdx_cpu_arr.emplace_back(dEdx);
    }

    _FLOAT cpu_elasped = cTimeUtil::End("cpu");

    auto net_gpu = mNet1dLstGPU[0];

    net_gpu->forward_unnormed_energy_grad_batch(batch_x);

    cTimeUtil::Begin("gpu");
    std::tuple<std::vector<float>, std::vector<tVectorXf>> ret =
        net_gpu->forward_unnormed_energy_grad_batch(batch_x);
    _FLOAT gpu_elasped = cTimeUtil::End("gpu");

    std::vector<float> E_gpu = std::get<0>(ret);
    std::vector<tVectorXf> dEdx_gpu = std::get<1>(ret);

    // _FLOAT E_gpu = net_gpu->forward_unnormed(x);

    for (int i = 0; i < test_n; i++)
    {
        printf("[%d] E cpu %.2f gpu %.2f\n", i, E_cpu_arr[i], E_gpu[i]);
    }
    for (int i = 0; i < test_n; i++)
    {
        printf("[%d] dEdx cpu %.2f gpu %.2f\n", i, dEdx_cpu_arr[i][0],
               dEdx_gpu[i][0]);
    }
    printf("[1d] cpu %.2f gpu %.2f\n", cpu_elasped, gpu_elasped);
}
int main()
{
    std::string path = "dnn.json";
    BuildNet(path);
    energy_grad_verify();
    // std::vector<int> cpu_val = {3, 3, 4, 5};
    // cCudaArray<int> gpu_val;
    // // gpu_val.Upload(cpu_val.data(), cpu_val.size());
    // gpu_val.Upload(cpu_val);

    // std::vector<int> cpu_val_new;
    // gpu_val.Download(cpu_val_new);
    // std::cout << cpu_val_new[0] << " " << cpu_val_new[1] << " "
    //           << cpu_val_new[2] << " " << cpu_val_new[3] << "\n";
    // // std::vector<float> cpu_val_new;
    // // gpu_val.Download(cpu_val_new);
    // // std::cout << cpu_val_new.size() << " " << cpu_val_new[0] << std::endl;
    // VisitStats(gpu_val);
}
