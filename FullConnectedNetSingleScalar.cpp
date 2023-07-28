#include "FullConnectedNetSingleScalar.h"
#include "utils/LogUtil.h"

template <typename dtype, int dim>
std::vector<dtype>
convert_eigenvec_to_stdvec(const Eigen::Matrix<dtype, dim, 1> &vec)
{
    std::vector<dtype> ret(vec.size());
    for (int i = 0; i < vec.size(); i++)
        ret[i] = vec[i];
    return ret;
}

template <typename dtype>
Eigen::Matrix<dtype, Eigen::Dynamic, 1>
convert_stdvec_to_eigenvec(const std::vector<dtype> &std_vec)
{
    int num = std_vec.size();
    Eigen::Matrix<dtype, Eigen::Dynamic, 1> x(num);
    for (int i = 0; i < std_vec.size(); i++)
    {
        x[i] = std_vec[i];
    }

    return x;
}
int cFCNetworkSingleScalar::GetNumOfParam() const
{
    int num_of_param = 0;
    for (auto &w : this->mWLst)
        num_of_param += w.size();
    for (auto &w : this->mbLst)
        num_of_param += w.size();
    return num_of_param;
}

template <typename dtype> int cFCNetworkSingleScalar::GetBytesOfParam() const
{
    return sizeof(dtype) * GetNumOfParam();
}

template <> int cFCNetworkSingleScalar::GetBytesOfParam<float>() const
{
    return sizeof(float) * GetNumOfParam();
}
tVectorX Softplus(const tVectorX &x)
{
    tVectorX y = (1.0 + x.array().exp()).log();
    return y;
}

tVectorX SoftplusGrad(const tVectorX &x)
{
    tVectorX y = x.array().exp() / (1 + x.array().exp());
    return y;
}

void cFCNetworkSingleScalar::Init(int input_dim_, int output_dim_, int layers_,
                                  const std::vector<tMatrixX> &weight_lst_,
                                  const std::vector<tVectorX> &bias_lst_,
                                  std::string act_, const tVectorX &input_mean,
                                  const _FLOAT &output_mean,
                                  const tVectorX &input_std,
                                  const _FLOAT &output_std)

{
    mInputDim = input_dim_;
    mOutputDim = output_dim_;
    mLayers.resize(layers_, 0);
    mWLst = weight_lst_;
    mbLst = bias_lst_;
    mActName = act_;
    mAct = Softplus;
    mActGrad = SoftplusGrad;

    mInputMean = input_mean;
    mOutputMean = output_mean;
    mInputStd = input_std;
    mOutputStd = output_std;

    SIM_ASSERT(mInputMean.size() == mInputDim);
    SIM_ASSERT(mInputStd.size() == mInputDim);
    // SIM_ASSERT(mOutputMean.size() == mOutputDim);
    // SIM_ASSERT(mOutputStd.size() == mOutputDim);
}
#include "utils/DefUtil.h"
#include "utils/FileUtil.h"
#include "utils/JsonUtil.h"
#include "utils/LogUtil.h"
#include <iostream>
#include <utils/json/json.h>
cFCNetworkSingleScalar::cFCNetworkSingleScalar() {}
void cFCNetworkSingleScalar::Init(const Json::Value &root)
{

    // parse value
    mInputDim = cJsonUtil::ParseAsInt("input_dim", root);
    mOutputDim = cJsonUtil::ParseAsInt("output_dim", root);
    SIM_ASSERT(mOutputDim == 1);
    mLayers =
        cJsonUtil::ReadVectorJson<int>(cJsonUtil::ParseAsValue("layers", root));

    mXRange = cJsonUtil::ReadMatrixJson<_FLOAT>("x_range", root, 2, mInputDim);

    // SIM_INFO("load DNN input range {}", mXRange);

    Json::Value weight_lst = cJsonUtil::ParseAsValue("weight_lst", root);
    Json::Value bias_lst = cJsonUtil::ParseAsValue("bias_lst", root);

    // get layer unit lst
    std::vector<int> layer_unit_lst = {};
    layer_unit_lst.push_back(mInputDim);
    for (int i = 0; i < mLayers.size(); i++)
    {
        layer_unit_lst.push_back(mLayers[i]);
    }
    layer_unit_lst.push_back(mOutputDim);

    SIM_ASSERT(weight_lst.size() == (layer_unit_lst.size() - 1));
    SIM_ASSERT(bias_lst.size() == (layer_unit_lst.size() - 1));

    // SIM_INFO("layer lst {}",
    //          convert_stdvec_to_eigenvec(layer_unit_lst).transpose());
    // parse network weight
    mWLst.clear();
    mbLst.clear();
    for (int i = 0; i < layer_unit_lst.size() - 1; i++)
    {
        int input = layer_unit_lst[i], output = layer_unit_lst[i + 1];
        // printf("======== layer %d, (%d->%d)========\n", i, input, output);
        tMatrixX w;
        Json::Value weight = weight_lst[i];
        cJsonUtil::ReadMatrixJson(weight, w);

        SIM_ASSERT((w.rows() == output) && (w.cols() == input));
        Json::Value bias = bias_lst[i];
        // printf("w = %d %d\n", w.rows(), w.cols());

        tVectorX b = cJsonUtil::ReadVectorJson<_FLOAT>(bias, output);

        mWLst.push_back(w);
        mbLst.push_back(b);
        // printf("b = %d\n", b.size());
    }

    mAct = Softplus;
    mActGrad = SoftplusGrad;

    // load statistics
    mInputMean =
        cJsonUtil::ReadVectorJson<_FLOAT>("input_mean", root, mInputDim);
    mInputStd = cJsonUtil::ReadVectorJson<_FLOAT>("input_std", root, mInputDim);
    mOutputMean = cJsonUtil::ReadVectorJson<_FLOAT>("output_mean", root, 1)[0];
    mOutputStd = cJsonUtil::ReadVectorJson<_FLOAT>("output_std", root, 1)[0];
    // SIM_INFO("input mean = {}", mInputMean.transpose());
    // SIM_INFO("output mean = {}", mOutputMean);
    // SIM_INFO("input std = {}", mInputStd.transpose());
    // SIM_INFO("output std = {}", mOutputStd);
}
void cFCNetworkSingleScalar::Init(std::string path)
{
    if (cFileUtil::ExistsFile(path) == false)
    {
        SIM_ERROR("fc network {} doesn't exist", path);
        exit(1);
    }
    SIM_INFO("load fc network from {}", path);
    Json::Value root;
    cJsonUtil::LoadJson(path, root);
    cFCNetworkSingleScalar::Init(root);
}

double Exp(double x) // the functor we want to apply
{
    return std::exp(x);
}

_FLOAT cFCNetworkSingleScalar::forward_normed(const tVectorX &x)
{
    // x : I,

    assert(x.size() == mInputDim);
    tVectorX z = x; // z : M,
    // std::cout << "input x(after normed) = " << x.transpose() << std::endl;
    for (int i = 0; i < mWLst.size(); i++)
    {
        // W: M x I, b : M x 1
        z = mWLst[i] * z; // z : M x N
        z += mbLst[i];
        // if(i == 0)
        // {
        //     std::cout << "w = \n" << mWLst[i] << std::endl;
        //     std::cout << "b = " << mbLst[i].transpose() << std::endl;
        // }
        // std::cout << "layer " << i << " z(before act) = " << z.transpose()
        //           << std::endl;
        // printf("i = %d z shape %d %d\n", i, z.rows(), z.cols());
        if (mAct != nullptr && i != mWLst.size() - 1)
        {
            z = mAct(z).eval();
            // std::cout << "layer " << i << " z(after act) = " << z.transpose()
            //           << std::endl;
        }
    }

    return z[0];
}

tVectorX cFCNetworkSingleScalar::calc_grad_wrt_input_normed(const tVectorX &x)
{
    assert(x.size() == mInputDim);
    tVectorX z = x;
    // tEigenArr<tVectorX> z_linear_lst = {};

    // Calculate gradient

    tVectorX grad_lst(mInputDim); // (N, O, I)
    grad_lst.setZero();

    // allocate cur_result
    tMatrixX cur_result;

    for (int layer_id = 0; layer_id < mWLst.size(); layer_id++)
    {
        z = mWLst[layer_id] * z;
        z += mbLst[layer_id];
        // std::cout << "layer " << layer_id << " z = " << z.transpose()
        //           << std::endl;
        // 1. compute for gradient
        {
            const tMatrixX &wi = mWLst[layer_id];
            // 1. allocate cur_multi
            tMatrixX cur_multi;
            if (mAct != nullptr && layer_id != mWLst.size() - 1)
            {
                // 2. row multication of matrix
                tVectorX dxi_next_d_zi = mActGrad(z);
                cur_multi.noalias() = dxi_next_d_zi.asDiagonal() * wi;
            }
            else
            {
                cur_multi.noalias() = wi;
            }
            // std::cout << "layer " << layer_id << " cur_multi = \n"
            //           << cur_multi << std::endl;
            // 2. multiply to calculate cur_result
            if (layer_id == 0)
            {
                cur_result.noalias() = cur_multi;
            }
            else
            {
                cur_result = cur_multi * cur_result;
            }
            // std::cout << "layer " << layer_id << " cur_res = \n"
            //           << cur_result << std::endl;
        }

        // compute for activation
        if (mAct != nullptr && layer_id != mWLst.size() - 1)
        {
            z = mAct(z);
        }
    }

    grad_lst.noalias() = cur_result.row(0);
    // grad_lst
    //     .slice(CreateTensorIndex<3>({i, 0, 0}),
    //            CreateTensorIndex<3>({1, mInputDim, mOutputDim}))
    //     .reshape(CreateTensorIndex<2>({mOutputDim, mInputDim})) =
    //     cur_result;

    return grad_lst;
}

tMatrixX cFCNetworkSingleScalar::calc_hess_wrt_input_normed(const tVectorX &x)
{
    int N = 1;
    tMatrixX ret(mInputDim, mInputDim);
    ret.setZero();

    /*
    x: I
    y: 1
    dydx: I (GradType)
    dy2dx2: I x I (HessType)
    */
    _FLOAT eps = 1e-3;
    // for (int data_id = 0; data_id < N; data_id++)
    {

        // 1. calc raw gra
        tVectorX cur_x = x;
        // SIM_ASSERT(cur_x.rows() == 1 && cur_x.cols() == mInputDim);

        tVectorX old_grad =
            this->calc_grad_wrt_input_normed(cur_x); // 1 x O x I

        for (int dim_id = 0; dim_id < mInputDim; dim_id++)
        {
            cur_x(dim_id) += eps;
            tVectorX new_grad = this->calc_grad_wrt_input_normed(cur_x);
            ret.col(dim_id) = (new_grad - old_grad) / eps; // 1 X O x I

            cur_x(dim_id) -= eps;
        }
    }
    return ret;
}
// tVectorX Net::Clamp(const tVectorX &x_raw)
// {
//     tMatrixX x_new = x_raw;

//     for (int i = 0; i < mInputDim; i++)
//     {
//         _FLOAT min = mXRange(0, i), max = mXRange(1, i);
//         _FLOAT size = max - min;
//         _FLOAT min_coef = x_new.col(i).minCoeff(),
//                max_coef = x_new.col(i).maxCoeff();
//         if (min_coef < min || max_coef > max)
//         {
//             printf("cur DNN input min %.2f max %.2f exceeds range [%.2f
//             %.2f], "
//                    "clamp\n",
//                    min_coef, max_coef, min, max);
//             x_new.col(i) = x_new.col(i).cwiseMin(max + 0.01 * size);
//             x_new.col(i) = x_new.col(i).cwiseMax(min - 0.01 * size);
//         }
//     }
//     return x_new;
// }
/*
    x_normed = (x - input_mean) / input_std
    x_normed -> DNN -> y_normed
    y_unnormed = y_normed * output_std + output_mean

*/
#include "utils/ProfUtil.h"
_FLOAT cFCNetworkSingleScalar::forward_unnormed(const tVectorX &x_unnormed)
{
    // cProfUtil::Begin("cloth/update_mat/cycles/forward");
    // tMatrixX x_unnormed = Clamp(x_unnormed_);
    // tMatrixX x_unnormed = x_unnormed_;

    // if (mInputMean.size() == 2)
    // {
    //     for (int j = 0; j < mInputDim; j++)
    //     {
    //         _FLOAT val = x_unnormed[j];
    //         _FLOAT min = mXRange(0, j), max = mXRange(1, j);
    //         if (val < min || val > max)
    //         {
    //             printf("[forwardE] cur x %.3f at data[%d] exceed range [% .3f
    //             "
    //                    "% .3f]\n",
    //                    val, j, min, max);
    //             SIM_WARN("input mean {} std {}", mInputMean.transpose(),
    //                      mInputStd.transpose());
    //             return 0;
    //         }
    //     }
    // }
    tVectorX x_normed =
        mInputStd.cwiseInverse().cwiseProduct(x_unnormed - mInputMean);
    // tMatrixX x_normed =
    //     (x_unnormed.rowwise() - mInputMean.transpose()).array().rowwise() /
    //     mInputStd.transpose().array();
    _FLOAT normed_y = forward_normed(x_normed);
    _FLOAT unnormed_y = (normed_y * mOutputStd) + mOutputMean;
    // tMatrixX unnormed_y =
    //     (normed_y.array().rowwise() * mOutputStd.transpose().array())
    //         .rowwise() +
    //     mOutputMean.transpose().array();
    // cProfUtil::End("cloth/update_mat/cycles/forward");
    return unnormed_y;
}
void cFCNetworkSingleScalar::SetComment(std::string) {}
tVectorX
cFCNetworkSingleScalar::calc_grad_wrt_input_unnormed(const tVectorX &x_unnormed)
{
    // int N = x_unnormed.rows();
    // if (mInputMean.size() == 2)
    // {
    //     for (int j = 0; j < mInputDim; j++)
    //     {
    //         _FLOAT val = x_unnormed[j];
    //         _FLOAT min = mXRange(0, j), max = mXRange(1, j);
    //         if (val < min || val > max)
    //         {
    //             // printf("[GradE] cur x %.3f at data(%d,%d) exceed range
    //             // [%.3f
    //             // %.3f]\n",
    //             //        val, i, j, min, max);
    //             // SIM_WARN("input mean {} std {}", mInputMean.transpose(),
    //             // mInputStd.transpose());
    //             return tVectorX::Zero(mInputMean.size());
    //         }
    //     }
    // }
    // // cProfUtil::Begin("cloth/update_mat/cycles/forward_grad");
    // tMatrixX x_unnormed = Clamp(x_unnormed_);

    // SIM_INFO("x_unnormed {}", x_unnormed);
    tVectorX x_normed =
        mInputStd.cwiseInverse().cwiseProduct(x_unnormed - mInputMean);
    // SIM_INFO("x_normed {}", x_normed);
    tVectorX normed_grad = this->calc_grad_wrt_input_normed(x_normed);
    // SIM_INFO("normed_grad = {}", normed_grad.transpose());
    tVectorX res =
        mInputStd.cwiseInverse().cwiseProduct(mOutputStd * normed_grad);
    // SIM_INFO("calc_grad_wrt_input_unnormed = {}", res.transpose());
    // SIM_INFO("mOutputStd = {}", mOutputStd);
    // SIM_INFO("mInputStd = {}", mInputStd.transpose());
    return res;
}
tMatrixX
cFCNetworkSingleScalar::calc_hess_wrt_input_unnormed(const tVectorX &x_unnormed)
{
    // cProfUtil::Begin("cloth/update_mat/cycles/forward_hess");
    // tMatrixX x_unnormed = Clamp(x_unnormed_);
    // tMatrixX x_unnormed = x_unnormed_;
    // int N = x_unnormed.rows();
    tMatrixX ret(mInputDim, mInputDim);
    ret.setZero();

    // int N = x_unnormed.rows();
    // if (mInputMean.size() == 2)
    // {
    //     for (int j = 0; j < mInputDim; j++)
    //     {
    //         _FLOAT val = x_unnormed[j];
    //         _FLOAT min = mXRange(0, j), max = mXRange(1, j);
    //         if (val < min || val > max)
    //         {
    //             // printf("[HessE] cur x %.3f at data(%d,%d) exceed range
    //             // [%.3f
    //             // %.3f]\n",
    //             //        val, i, j, min, max);
    //             // SIM_WARN("input mean {} std {}", mInputMean.transpose(),
    //             // mInputStd.transpose());
    //             return ret;
    //         }
    //     }
    // }

    /*
    x: N x I
    y: N x O
    dydx: N x O x I (GradType)
    dy2dx2: N x O x I x I (HessType)
    */
    _FLOAT eps = 1e-3;

    // 1. calc raw gra

    tVectorX cur_x = x_unnormed;
    tVectorX old_grad = this->calc_grad_wrt_input_unnormed(cur_x); // 1 x O x I

    for (int dim_id = 0; dim_id < mInputDim; dim_id++)
    {
        cur_x(dim_id) += eps;
        tVectorX new_grad = this->calc_grad_wrt_input_unnormed(cur_x);
        ret.col(dim_id) = (new_grad - old_grad) / eps; // 1 X O x I
        cur_x(dim_id) -= eps;
    }

    // cProfUtil::End("cloth/update_mat/cycles/forward_hess");
    return ret;
}

bool cFCNetworkSingleScalar::CheckUnnormedInputExceedRangeAndComplain(
    const tVectorX &x_unnormed, std::string label, bool silence) const
{
    for (int j = 0; j < mInputDim; j++)
    {
        _FLOAT val = x_unnormed[j];
        _FLOAT min = mXRange(0, j), max = mXRange(1, j);
        if (val < min || val > max)
        {
            if (silence == false)
            {
                printf("[HessE] cur x %.3f at data[%d] exceed range [%.3f %"
                       ".3f]\n ",
                       val, j, min, max);
                SIM_WARN("input mean {} std {}", mInputMean.transpose(),
                         mInputStd.transpose());
            }
            return true;
        }
    }
    return false;
}

void cFCNetworkSingleScalar::SetBakedInfo(tNetBakeInfoPtr ptr)
{
    mBakeInfo = ptr;
}

int findInterval1D(const float x, float xmin, float xmax, int N)
{
    if (x < xmin || x > xmax)
    {
        return -1; // x is out of the range of x_arr
    }

    float step = (xmax - xmin) / (N - 1);

    int x_index = std::floor((x - xmin) / step);
    return x_index; // x is in the interval (x_arr[idx-1], x_arr[idx])
}
tVector2i findInterval2D(const tVectorXf &x, float xmin, float xmax, float ymin,
                         float ymax, int N)
{
    // Calculate the steps
    float xstep = (xmax - xmin) / (N - 1);
    float ystep = (ymax - ymin) / (N - 1);

    // Compute the indices
    int x_index = std::floor((x.x() - xmin) / xstep);
    int y_index = std::floor((x.y() - ymin) / ystep);

    // Check if x is within the grid
    if (x_index < 0 || x_index >= N - 1 || y_index < 0 || y_index >= N - 1)
    {
        return tVector2i(-1, -1); // x is out of the range of the grid
    }

    return tVector2i(x_index, y_index);
}
typedef Eigen::Matrix<float, 4, 1> tVector4f;
tVector4f CalcBilinearInterpolationCoef(float xmin, float xmax, float ymin,
                                        float ymax, float x, float y)
{
    float x_frac = (x - xmin) / (xmax - xmin);
    float y_frac = (y - ymin) / (ymax - ymin);

    float x_coeff1 = 1.0f - x_frac;
    float x_coeff2 = x_frac;
    float y_coeff1 = 1.0f - y_frac;
    float y_coeff2 = y_frac;

    return tVector4f(x_coeff1 * y_coeff1, x_coeff2 * y_coeff1,
                     x_coeff1 * y_coeff2, x_coeff2 * y_coeff2);
}

void cFCNetworkSingleScalar::calc_E_grad_hess_unnormed_baked(
    const tVectorX &x, float &final_E, tVectorXf &final_grad,
    tMatrixXf &final_hess)
{
    if (x.size() == 1)
    {
        // 1D
        int idx = findInterval1D(x[0], this->mXRange(0, 0), this->mXRange(1, 0),
                                 mBakeInfo->samples);
        SIM_ASSERT(idx >= 0 && idx < mBakeInfo->samples);

        float xmin = mBakeInfo->x_arr[idx][0],
              xmax = mBakeInfo->x_arr[idx + 1][0];

        final_E = 0;
        final_grad.noalias() = tVectorXf::Zero(1);
        final_hess.noalias() = tMatrixXf::Zero(1, 1);

        float coef = (x[0] - xmin) / (xmax - xmin);

        tVectorXf x_pred = (1 - coef) * mBakeInfo->x_arr[idx] +
                           coef * mBakeInfo->x_arr[idx + 1];
        SIM_INFO("x_pred = {}, x_input = {}", x_pred.transpose(),
                 x.transpose());
        final_E = (1 - coef) * mBakeInfo->e_arr[idx] +
                  coef * mBakeInfo->e_arr[idx + 1];
        final_grad.noalias() = (1 - coef) * mBakeInfo->grad_arr[idx] +
                               coef * mBakeInfo->grad_arr[idx + 1];
        final_hess.noalias() = (1 - coef) * mBakeInfo->hess_arr[idx] +
                               coef * mBakeInfo->hess_arr[idx + 1];
    }
    else
    {
        // 2D

        float xmin = mXRange(0, 0), xmax = mXRange(1, 0), ymin = mXRange(0, 1),
              ymax = mXRange(1, 1);
        tVector2i xmin_ymin_idx =
            findInterval2D(x.cast<float>(), xmin, xmax, ymin, ymax, mBakeInfo->samples);
        
        SIM_ASSERT(xmin_ymin_idx.minCoeff() > 0);

        auto visit = [](const tVector2i &x, int samples) -> int
        { return x[0] * samples + x[1]; };

        tVector2i xmax_ymax_idx = xmin_ymin_idx + tVector2i::Ones();

        int indices[4] = {
            visit(xmin_ymin_idx, mBakeInfo->samples),
            visit(xmin_ymin_idx + tVector2i(1, 0), mBakeInfo->samples),
            visit(xmin_ymin_idx + tVector2i(0, 1), mBakeInfo->samples),
            visit(xmax_ymax_idx, mBakeInfo->samples),
        };
        tVector2f xmin_ymin_x = mBakeInfo->x_arr[indices[0]];
        tVector2f xmax_ymax_x = mBakeInfo->x_arr[indices[3]];

        SIM_ASSERT(x[0] >= xmin_ymin_x[0] && x[0] <= xmax_ymax_x[0]);
        SIM_ASSERT(x[1] >= xmin_ymin_x[1] && x[1] <= xmax_ymax_x[1]);

        tVector4f coef = CalcBilinearInterpolationCoef(
            xmin_ymin_x[0], xmax_ymax_x[0], xmin_ymin_x[1], xmax_ymax_x[1],
            x[0], x[1]);

        tVectorXf x_pred = tVectorXf::Zero(2);

        // combine for E, dEdx, E_hess
        final_E = 0;
        final_grad.noalias() = tVectorXf::Zero(2);
        final_hess.noalias() = tMatrixXf::Zero(2, 2);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            x_pred += coef[i] * mBakeInfo->x_arr[indices[i]];
            final_E += coef[i] * mBakeInfo->e_arr[indices[i]];
            final_grad += coef[i] * mBakeInfo->grad_arr[indices[i]];
            final_hess += coef[i] * mBakeInfo->hess_arr[indices[i]];
        }
        SIM_INFO("x_pred = {}, x_input = {}", x_pred.transpose(),
                 x.transpose());
    }
}