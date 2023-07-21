#pragma once
#include "utils/DefUtil.h"
#include "utils/EigenUtil.h"
#include "utils/TensorUtil.h"

namespace Json
{
class Value;
};
class Net
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit Net();
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
    virtual void SetComment(std::string comment);
    virtual std::string GetComment() const { return this->mComment; }
    int mInputDim;
    int mOutputDim;
    virtual bool
    CheckUnnormedInputExceedRangeAndComplain(const tVectorX &x_unnormed,
                                             std::string label,
                                             bool silence = false) const;

protected:
    std::string mComment;
    tVectorXi mLayers;
    std::vector<tMatrixX> mWLst;
    std::vector<tVectorX> mbLst;
    std::string mActName;
    std::function<tVectorX(tVectorX)> mAct, mActGrad;
    tMatrixX mXRange; // [2, I]. the first row is min, the second row is max
    tVectorX mInputMean, mInputStd;
    _FLOAT mOutputMean, mOutputStd;
    tMatrixX Clamp(const tMatrixX &);
};
SIM_DECLARE_PTR(Net);