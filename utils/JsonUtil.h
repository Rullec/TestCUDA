#pragma once
#include "utils/EigenUtil.h"
#include "utils/SparseUtil.h"
#include <string>

namespace Json
{
class Value;
};
class cJsonUtil
{
public:
    // static Json::Value BuildVectorJson(const tVector &vec);
    template <typename dtype>
    static Eigen::Matrix<dtype, Eigen::Dynamic, 1>
    ReadVectorJson(const Json::Value &root, int requested_size = -1);

    template <typename dtype>
    static Eigen::Matrix<dtype, Eigen::Dynamic, 1>
    ReadVectorJson(std::string key, const Json::Value &root,
                   int requested_size = -1);
    static tVectorX ReadVectorJsonFromFile(const std::string &path,
                                           const Json::Value &root);
    template <typename dtype>
    static Json::Value
    BuildVectorJson(const Eigen::Matrix<dtype, Eigen::Dynamic, 1> &vec);
    static std::string BuildVectorString(const tVectorX &vec);
    static Json::Value BuildSparseMatJson(const tSparseMat &mat);
    static Json::Value BuildSparseMatJson(int rows, int cols,
                                          const tEigenArr<tTriplet> &entrys);
    static tEigenArr<tTriplet> ParseSparseMat(const Json::Value &value,
                                              int &rows, int &cols);
    // static bool ReadVectorJson(const Json::Value &root,
    //                            tVectorX &out_vec);
    // static bool ReadVectorJson(const Json::Value &root, tVector4 &out_vec);

    template <typename dtype>
    static Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic>
    ReadMatrixJson(std::string key, const Json::Value &root, int rows = -1,
                   int cols = -1);
    static bool ReadMatrixJson(const Json::Value &root, tMatrixX &out_mat);
    static bool ReadMatrixJson(const Json::Value &root, tMatrixXi &out_mat);
    static bool LoadJson(const std::string &path, Json::Value &value);
    static bool LoadJsonFromString(const std::string &path, Json::Value &value);
    static bool WriteJson(const std::string &path, Json::Value &value,
                          bool indent = true, int precision = 5);
    static std::string WriteJson2Str(Json::Value &value, bool indent = true,
                                     int precision = 5);
    static int ParseAsInt(const std::string &data_field_name,
                          const Json::Value &root);
    static std::string ParseAsString(const std::string &data_field_name,
                                     const Json::Value &root);
    static _FLOAT ParseAsfloat(const std::string &data_field_name,
                               const Json::Value &root);
    static _FLOAT ParseAsFloat(const std::string &data_field_name,
                               const Json::Value &root);
    static bool ParseAsBool(const std::string &data_field_name,
                            const Json::Value &root);
    static Json::Value ParseAsValue(const std::string &data_field_name,
                                    const Json::Value &root);
    static bool HasValue(const std::string &name, const Json::Value &root);
};
