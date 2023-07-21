
#include "utils/EigenUtil.h"

_FLOAT EvalBicubicHermite(const tVector4 &fval_lst, const tVector4 &fx_lst,
                          const tVector4 &fy_lst, const tVector4 &fxy_lst,
                          const tVector2 &x_range, const tVector2 &y_range,
                          _FLOAT X_, _FLOAT Y_);

std::vector<_FLOAT> EvalPiecewiseBicubicHermte(
    const std::vector<_FLOAT> &x_lst, const std::vector<_FLOAT> &y_lst,
    const std::vector<_FLOAT> &fval_lst, const std::vector<_FLOAT> &fx_lst,
    const std::vector<_FLOAT> &fy_lst, const std::vector<_FLOAT> &fxy_lst,
    const std::vector<_FLOAT> &x_sample_lst,
    const std::vector<_FLOAT> &y_sample_lst);