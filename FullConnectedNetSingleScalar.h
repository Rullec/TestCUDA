#pragma once
#include "utils/DefUtil.h"
#include "utils/EigenUtil.h"
#include "utils/TensorUtil.h"

namespace Json
{
class Value;
};

typedef Eigen::Matrix<float, 2, 2> tMatrix2f;
typedef Eigen::Matrix<float, 2, 1> tVector2f;

struct tNetBakeInfo
{
    int samples;
    std::vector<tVectorXf> x_arr = {};
    std::vector<float> e_arr = {};
    std::vector<tVectorXf> grad_arr = {};
    std::vector<tMatrixXf> hess_arr = {};
};
SIM_DECLARE_STRUCT_AND_PTR(tNetBakeInfo);

class cFCNetworkSingleScalar
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit cFCNetworkSingleScalar();
    virtual void Init(std::string path);
    virtual void Init(const Json::Value &path);
    virtual void Init(int input_dim, int output_dim, int layers,
                      const std::vector<tMatrixX> &weight_lst,
                      const std::vector<tVectorX> &bias_lst, std::string act,
                      const tVectorX &input_mean, const _FLOAT &output_mean,
                      const tVectorX &input_std, const _FLOAT &output_std);

    int GetNumOfParam() const;
    template <typename dtype> int GetBytesOfParam() const;
    // ============== input & output are all normed ===============
    virtual _FLOAT forward_normed(const tVectorX &x);
    virtual tVectorX calc_grad_wrt_input_normed(const tVectorX &x);
    virtual tMatrixX calc_hess_wrt_input_normed(const tVectorX &x);

    // ============== input & output are all unnormed ===============
    virtual _FLOAT forward_unnormed(const tVectorX &x);
    virtual tVectorX calc_grad_wrt_input_unnormed(const tVectorX &x);
    virtual tMatrixX calc_hess_wrt_input_unnormed(const tVectorX &x);
    virtual void calc_E_grad_hess_unnormed_baked(const tVectorX &x, float &E,
                                                 tVectorXf &grad,
                                                 tMatrixXf &hess);

    virtual void SetComment(std::string comment);
    virtual std::string GetComment() const { return this->mComment; }
    int mInputDim;
    int mOutputDim;
    virtual bool
    CheckUnnormedInputExceedRangeAndComplain(const tVectorX &x_unnormed,
                                             std::string label,
                                             bool silence = false) const;
    tMatrixX mXRange; // [2, I]. the first row is min, the second row is max
    void SetBakedInfo(tNetBakeInfoPtr ptr);

protected:
    std::string mComment;
    tVectorXi mLayers;
    std::vector<tMatrixX> mWLst;
    std::vector<tVectorX> mbLst;
    std::string mActName;
    std::function<tVectorX(tVectorX)> mAct, mActGrad;

    tVectorX mInputMean, mInputStd;
    _FLOAT mOutputMean, mOutputStd;
    tMatrixX Clamp(const tMatrixX &);

    // ========= bakes =========
    tNetBakeInfoPtr mBakeInfo;
    // bool mHaveBakedInfo;
    // tMatrixXf mBakedX;
    // tVectorXf mBasked_E;
    // tMatrixXf mBasked_grad;
    // std::vector<tMatrixXf> mBasked_hess;
};

SIM_DECLARE_PTR(cFCNetworkSingleScalar);