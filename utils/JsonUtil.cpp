#include "JsonUtil.h"
#include "FileUtil.h"
#include "LogUtil.h"
#include "utils/DefUtil.h"
#include "json/json.h"
#include <fstream>
#include <iostream>
#include <memory>

// tLogger cJsonUtil::mLogger = cLogUtil::CreateLogger("cJsonUtil");
template <typename dtype>
Json::Value
cJsonUtil::BuildVectorJson(const Eigen::Matrix<dtype, Eigen::Dynamic, 1> &vec)
{
    Json::Value json = Json::arrayValue;

    for (int i = 0; i < vec.size(); ++i)
    {
        json.append(vec[i]);
    }
    return json;
}
template Json::Value cJsonUtil::BuildVectorJson<_FLOAT>(
    const Eigen::Matrix<_FLOAT, Eigen::Dynamic, 1> &vec);
template Json::Value cJsonUtil::BuildVectorJson<int>(
    const Eigen::Matrix<int, Eigen::Dynamic, 1> &vec);
template Json::Value cJsonUtil::BuildVectorJson<float>(
    const Eigen::Matrix<float, Eigen::Dynamic, 1> &vec);
template Json::Value cJsonUtil::BuildVectorJson<unsigned int>(
    const Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> &vec);

std::string cJsonUtil::BuildVectorString(const tVectorX &vec)
{
    std::string str = "";
    char str_buffer[32];
    for (int i = 0; i < vec.size(); ++i)
    {
        if (i != 0)
        {
            str += ",";
        }
        sprintf(str_buffer, "%20.10f", vec[i]);
        str += std::string(str_buffer);
    }
    return str;
}

bool cJsonUtil::LoadJson(const std::string &path, Json::Value &value)
{
    // cFileUtil::AddLock(path);
    // std::cout <<"parsing " << path << " begin \n";
    std::ifstream fin(path);
    if (fin.fail() == true)
    {
        std::cout << "[error] cJsonUtil::LoadJson file " << path
                  << " doesn't exist\n";
        return false;
    }
    Json::CharReaderBuilder rbuilder;
    std::string errs;
    bool parsingSuccessful =
        Json::parseFromStream(rbuilder, fin, &value, &errs);
    if (!parsingSuccessful)
    {
        // report to the user the failure and their locations in the
        // document.
        std::cout << "[error] cJsonUtil::LoadJson: Failed to parse json\n"
                  << errs << ", file path " << path << std::endl;
        return false;
    }
    // std::cout <<"parsing " << path << " end \n";
    // cFileUtil::DeleteLock(path);
    return true;
}
bool cJsonUtil::LoadJsonFromString(const std::string &content,
                                   Json::Value &value)
{
    Json::CharReaderBuilder builder;
    std::string errs;

    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    bool success = reader->parse(
        content.c_str(), content.c_str() + content.size(), &value, &errs);

    if (!success)
    {
        SIM_ERROR("Failed to parse JSON: ", errs);
        return false;
    }
    return true;
}
bool cJsonUtil::WriteJson(const std::string &path, Json::Value &value,
                          bool indent /* = true*/, int precision /*=5*/)
{
    std::string output_str = cJsonUtil::WriteJson2Str(value, indent, precision);
    auto dir = cFileUtil::GetDir(path);
    bool has_dir = dir.size() == 0 ? true : cFileUtil::ExistsDir(dir);
    if (has_dir == false)
    {
        SIM_ASSERT(cFileUtil::CreateDir(dir.c_str()));
    }

    std::ofstream fout(path);
    if (fout.fail() == true)
    {
        SIM_ERROR("WriteJson open {} failed", path);
        exit(1);
    }
    else
    {
        fout << output_str;
        fout.close();
    }
    // cFileUtil::DeleteLock(path);
    return fout.fail() == false;
}
std::string cJsonUtil::WriteJson2Str(Json::Value &value, bool indent,
                                     int precision)
{
    Json::StreamWriterBuilder builder;

    if (indent)
    {
        builder.settings_["indentation"] = "  ";
    }
    else
    {
        builder.settings_["indentation"] = "";
    }

    builder.settings_["precision"] = precision;
    std::string output = Json::writeString(builder, value);
    return output;
}

#define JSONUTIL_ASSERT_NULL(root, data) (root.isMember(data))

int cJsonUtil::ParseAsInt(const std::string &data_field_name,
                          const Json::Value &root)
{
    if (false == JSONUTIL_ASSERT_NULL(root, data_field_name))
    {
        SIM_ERROR("ParseAsInt {} failed", data_field_name.c_str());
        exit(0);
    }
    return root[data_field_name].asInt();
}

std::string cJsonUtil::ParseAsString(const std::string &data_field_name,
                                     const Json::Value &root)
{
    if (false == JSONUTIL_ASSERT_NULL(root, data_field_name))
    {
        SIM_ERROR("ParseAsString {} failed", data_field_name.c_str());
        exit(0);
    }
    return root[data_field_name].asString();
}

_FLOAT cJsonUtil::ParseAsfloat(const std::string &data_field_name,
                               const Json::Value &root)
{
    if (false == JSONUTIL_ASSERT_NULL(root, data_field_name))
    {
        SIM_ERROR("ParseAsfloat {} failed", data_field_name.c_str());
        exit(0);
    }
    return root[data_field_name].asFloat();
}

_FLOAT cJsonUtil::ParseAsFloat(const std::string &data_field_name,
                               const Json::Value &root)
{
    if (false == JSONUTIL_ASSERT_NULL(root, data_field_name))
    {
        SIM_ERROR("ParseAsFloat {} failed", data_field_name.c_str());
        exit(0);
    }
    return root[data_field_name].asFloat();
}

bool cJsonUtil::ParseAsBool(const std::string &data_field_name,
                            const Json::Value &root)
{
    if (false == JSONUTIL_ASSERT_NULL(root, data_field_name))
    {
        SIM_ERROR("ParseAsBool {} failed", data_field_name.c_str());
        exit(0);
    }
    return root[data_field_name].asBool();
}

Json::Value cJsonUtil::ParseAsValue(const std::string &data_field_name,
                                    const Json::Value &root)
{
    if (false == JSONUTIL_ASSERT_NULL(root, data_field_name))
    {
        SIM_ERROR("ParseAsValue {} failed", data_field_name.c_str());
        exit(0);
    }
    return root[data_field_name];
}

/**
 * \brief           read matrix json
 */
bool cJsonUtil::ReadMatrixJson(const Json::Value &root, tMatrixX &out_mat)
{
    out_mat.resize(0, 0);
    tEigenArr<tVectorX> mat(0);
    int num_of_cols = -1;
    for (int i = 0; i < root.size(); i++)
    {
        tVectorX res;
        res = cJsonUtil::ReadVectorJson<_FLOAT>(root[i]);

        if (num_of_cols == -1)
        {
            num_of_cols = res.size();
            mat.push_back(res);
        }
        else
        {
            // the dimension doesn't meet
            if (num_of_cols != res.size())
            {
                return false;
            }
            else
            {
                mat.push_back(res);
            }
        }
    }
    out_mat.noalias() = tMatrixX::Zero(mat.size(), num_of_cols);
    for (int i = 0; i < mat.size(); i++)
    {
        out_mat.row(i) = mat[i].transpose();
    }
    return true;
}

/**
 * \brief           read matrix json
 */
bool cJsonUtil::ReadMatrixJson(const Json::Value &root, tMatrixXi &out_mat)
{
    out_mat.resize(0, 0);
    tEigenArr<tVectorXi> mat(0);
    int num_of_cols = -1;
    for (int i = 0; i < root.size(); i++)
    {
        tVectorXi res = cJsonUtil::ReadVectorJson<_FLOAT>(root[i]).cast<int>();

        if (num_of_cols == -1)
        {
            num_of_cols = res.size();
            mat.push_back(res);
        }
        else
        {
            // the dimension doesn't meet
            if (num_of_cols != res.size())
            {
                return false;
            }
            else
            {
                mat.push_back(res);
            }
        }
    }
    out_mat.noalias() = tMatrixXi::Zero(mat.size(), num_of_cols);
    for (int i = 0; i < mat.size(); i++)
    {
        out_mat.row(i) = mat[i].transpose();
    }
    return true;
}

bool cJsonUtil::HasValue(const std::string &name, const Json::Value &root)
{
    return JSONUTIL_ASSERT_NULL(root, name);
}

template <typename dtype>
Eigen::Matrix<dtype, Eigen::Dynamic, 1>
cJsonUtil::ReadVectorJson(const Json::Value &val, int size)
{
    using vec = Eigen::Matrix<dtype, Eigen::Dynamic, 1>;
    vec ret = vec::Zero(val.size());
    for (int i = 0; i < ret.size(); i++)
    {
        ret[i] = val[i].asDouble();
    }
    if (size != -1 && size != ret.size())
    {
        SIM_ERROR("requested size {} but real size {}, invalid!", size,
                  ret.size());
    }
    return ret;
}

template tVectorX cJsonUtil::ReadVectorJson<_FLOAT>(const Json::Value &val,
                                                    int size = -1);
template tVectorXi cJsonUtil::ReadVectorJson<int>(const Json::Value &val,
                                                  int size = -1);
template Eigen::Matrix<unsigned int, Eigen::Dynamic, 1>
cJsonUtil::ReadVectorJson<unsigned int>(const Json::Value &val, int size = -1);

template <typename dtype>
Eigen::Matrix<dtype, Eigen::Dynamic, 1>
cJsonUtil::ReadVectorJson(std::string key, const Json::Value &root,
                          int requested_size)
{
    Eigen::Matrix<dtype, Eigen::Dynamic, 1> ret =
        cJsonUtil::ReadVectorJson<dtype>(cJsonUtil::ParseAsValue(key, root));

    if (requested_size != -1)
    {
        if (ret.size() != requested_size)
        {
            SIM_ERROR("when parsing {}, requested size {} != real size {}", key,
                      requested_size, ret.size());
        }
    }
    return ret;
}

template tVectorXf cJsonUtil::ReadVectorJson<float>(std::string key,
                                                    const Json::Value &root,
                                                    int requested_size = -1);
template tVectorX cJsonUtil::ReadVectorJson<_FLOAT>(std::string key,
                                                    const Json::Value &root,
                                                    int requested_size = -1);
template tVectorXi cJsonUtil::ReadVectorJson<int>(std::string key,
                                                  const Json::Value &root,
                                                  int requested_size = -1);

template <typename dtype>
Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic>
cJsonUtil::ReadMatrixJson(std::string key, const Json::Value &root, int rows,
                          int cols)
{
    Json::Value val = cJsonUtil::ParseAsValue(key, root);
    tMatrixX result;
    cJsonUtil::ReadMatrixJson(val, result);
    Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic> ret =
        result.cast<dtype>();
    if (rows != -1 && ret.rows() != rows)
    {
        SIM_ERROR("target rows {} != real rows {}", ret.rows(), rows);
    }

    if (cols != -1 && ret.cols() != cols)
    {
        SIM_ERROR("target col {} != real col {}", ret.cols(), cols);
    }
    return ret;
}

template tMatrixXi cJsonUtil::ReadMatrixJson<int>(std::string key,
                                                  const Json::Value &root,
                                                  int rows = -1, int cols = -1);
template tMatrixX cJsonUtil::ReadMatrixJson<_FLOAT>(std::string key,
                                                    const Json::Value &root,
                                                    int rows = -1,
                                                    int cols = -1);

Json::Value cJsonUtil::BuildSparseMatJson(const tSparseMat &A)
{
    Json::Value matrix;
    matrix["rows"] = static_cast<int>(A.rows());
    matrix["cols"] = static_cast<int>(A.cols());
    matrix["nonzeros"] = static_cast<int>(A.nonZeros());

    Json::Value coeffs(Json::arrayValue);
    for (int k = 0; k < A.outerSize(); ++k)
    {
        for (tSparseMat::InnerIterator it(A, k); it; ++it)
        {
            Json::Value coeff;
            coeff["row"] = static_cast<int>(it.row());
            coeff["col"] = static_cast<int>(it.col());
            coeff["value"] = it.value();
            coeffs.append(coeff);
        }
    }
    matrix["coeffs"] = coeffs;
    return matrix;
}

Json::Value cJsonUtil::BuildSparseMatJson(int rows, int cols,
                                          const tEigenArr<tTriplet> &entrys)
{
    Json::Value matrix;
    matrix["rows"] = rows;
    matrix["cols"] = cols;
    matrix["nonzeros"] = static_cast<int>(entrys.size());

    Json::Value coeffs(Json::arrayValue);
    for (auto &it : entrys)
    {
        Json::Value coeff;
        coeff["row"] = it.row();
        coeff["col"] = it.col();
        coeff["value"] = it.value();
        coeffs.append(coeff);
    }
    matrix["coeffs"] = coeffs;
    return matrix;
}

tEigenArr<tTriplet> cJsonUtil::ParseSparseMat(const Json::Value &value,
                                              int &rows, int &cols)
{
    rows = value["rows"].asInt();
    cols = value["cols"].asInt();
    int nzo = value["nonzeros"].asInt();
    tEigenArr<tTriplet> ret = {};
    for (int i = 0; i < nzo; i++)
    {
        auto &val = value["coeffs"][i];

        ret.emplace_back(val["row"].asInt(), val["int"].asInt(),
                         val["value"].asDouble());
    }
    return ret;
}