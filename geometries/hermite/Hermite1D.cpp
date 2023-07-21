#include "utils/DefUtil.h"
#include "utils/MathUtil.h"
#include "utils/TimeUtil.hpp"
void EvalHermite1D(_FLOAT x0, _FLOAT x1, _FLOAT f0, _FLOAT f1, _FLOAT d0,
                   _FLOAT d1, _FLOAT x, _FLOAT &f, _FLOAT &dfdx)
{
    /*
          p(x) = f_0 H_1(x) + f_1 H_2(x) + d_0 H_3(x) + d_1 H_4(x)
          H_1(x) = \phi(\frac{x_1 - x}{h})
          H_2(x) = \phi(\frac{x - x_1}{h})
          H_3(x) = - h \psi(\frac{x_1 - x}{h})
          H_4(x) = h \psi(\frac{x - x_0}{h})
          -----------------------
          \phi(t) = 3t^2 - 2t^3 \\
          \psi(t) = t^3 - t^2 \\
          h = x_1 - x_0
          */
    x = cMathUtil::Clamp(x, x0, x1);
    // assert x0 <= x and x <= x1
    auto phi = [](_FLOAT t) { return 3 * t * t - 2 * (t * t * t); };
    auto psi = [](_FLOAT t) { return t * t * t - t * t; };

    auto dphidt = [](_FLOAT t) { return 6 * t - 6 * t * t; };
    auto dpsidt = [](_FLOAT t) { return 3 * t * t - 2 * t; };

    _FLOAT h = x1 - x0;
    _FLOAT H1 = phi((x1 - x) / h);
    _FLOAT H2 = phi((x - x0) / h);
    _FLOAT H3 = -h * psi((x1 - x) / h);
    _FLOAT H4 = h * psi((x - x0) / h);

    f = f0 * H1 + f1 * H2 + d0 * H3 + d1 * H4;

    {
        _FLOAT dH1dt = -dphidt((x1 - x) / h) / h;
        _FLOAT dH2dt = dphidt((x - x0) / h) / h;
        _FLOAT dH3dt = dpsidt((x1 - x) / h);
        _FLOAT dH4dt = dpsidt((x - x0) / h);
        dfdx = f0 * dH1dt + f1 * dH2dt + d0 * dH3dt + d1 * dH4dt;
    }
}

void EvalPiecewiseHermite1D(const std::vector<_FLOAT> &t_lst,
                            const std::vector<_FLOAT> &q_lst,
                            const std::vector<_FLOAT> &m_lst, const _FLOAT x,
                            _FLOAT &fx, _FLOAT &dfdx)
{
    /*
        given a list of cubic hermite coeff, calculate the funcation value at x.
        q_lst: hermite func vals
        m_lst: hermite gradient vals
    */

    // assert np.min(t_lst) <= np.min(x_lst) and np.max(t_lst) >= np.max(x_lst)

    // id \in[0, len(t_lst) - 2];

    int id =
        std::lower_bound(t_lst.begin(), t_lst.end(), x) - t_lst.begin() - 1;
    id = SIM_MIN(SIM_MAX(0, id), t_lst.size() - 2);

    EvalHermite1D(t_lst[id], t_lst[id + 1], q_lst[id], q_lst[id + 1], m_lst[id],
                  m_lst[id + 1], x, fx, dfdx);
}

void EvalPiecewiseHermite1DGroup(const std::vector<_FLOAT> &t_lst,
                                 const std::vector<_FLOAT> &q_lst,
                                 const std::vector<_FLOAT> &m_lst,
                                 const std::vector<_FLOAT> x_array,
                                 std::vector<_FLOAT> &fx_array,
                                 std::vector<_FLOAT> &dfdx_array)
{

    int id = std::lower_bound(t_lst.begin(), t_lst.end(), x_array[0]) -
             t_lst.begin() - 1;
    id = SIM_MIN(SIM_MAX(0, id), t_lst.size() - 2);

    int N = x_array.size();
    fx_array.resize(N);
    dfdx_array.resize(N);
    for (int i = 0; i < x_array.size(); i++)
    {
        EvalHermite1D(t_lst[id], t_lst[id + 1], q_lst[id], q_lst[id + 1],
                      m_lst[id], m_lst[id + 1], x_array[i], fx_array[i],
                      dfdx_array[i]);
    }
}