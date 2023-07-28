#include "FullConnectedNetSingleScalar.h"
#include "FullConnectedNetSingleScalarGPU.h"
#include "utils/FileUtil.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include "utils/json/json.h"

template <typename dtype>
int GetOffsetFromIdx(const std::vector<dtype> &arr, const dtype &val)
{
    return std::find(arr.begin(), arr.end(), val) - arr.begin();
}

std::vector<cFCNetworkSingleScalarPtr> mNet1dLstCPU, mNet2dLstCPU;
std::vector<cFCNetworkSingleScalarGPUPtr> mNet1dLstGPU, mNet2dLstGPU;
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

        mNet1dLstCPU[offset_1d] = std::make_shared<cFCNetworkSingleScalar>();
        mNet1dLstCPU[offset_1d]->Init(net_val);
        mNet1dLstCPU[offset_1d]->SetComment(
            cJsonUtil::WriteJson2Str(comment_json));

        mNet1dLstGPU[offset_1d] = std::make_shared<cFCNetworkSingleScalarGPU>();
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

        mNet2dLstCPU[offset_2d] = std::make_shared<cFCNetworkSingleScalar>();
        mNet2dLstCPU[offset_2d]->Init(net_val);
        mNet2dLstCPU[offset_2d]->SetComment(
            cJsonUtil::WriteJson2Str(comment_json));

        mNet2dLstGPU[offset_2d] = std::make_shared<cFCNetworkSingleScalarGPU>();
        mNet2dLstGPU[offset_2d]->Init(net_val);
        mNet2dLstGPU[offset_2d]->SetComment(
            cJsonUtil::WriteJson2Str(comment_json));
        printf("[info] Load DNN dof %d %d from %s\n", idx_2d[0], idx_2d[1],
               cFileUtil::GetFilename(path).c_str());
    }
}
