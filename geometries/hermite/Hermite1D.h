#include "utils/BaseTypeUtil.h"
#include "utils/DefUtil.h"
#include <vector>

_FLOAT EvalHermite1D(_FLOAT x0, _FLOAT x1, _FLOAT f0, _FLOAT f1, _FLOAT d0,
                     _FLOAT d1, _FLOAT x);

std::vector<_FLOAT> EvalPiecewiseHermite1D(const std::vector<_FLOAT> &t_lst,
                                           const std::vector<_FLOAT> &q_lst,
                                           const std::vector<_FLOAT> &m_lst,
                                           const std::vector<_FLOAT> &x_lst);