#include "utils/EigenUtil.h"
#include "utils/LogUtil.h"
#include "utils/MathUtil.h"
#include <iostream>
/*
Given triangle vertices: v0, v1, v2

n = (v1 - v0).cross(v2 - v1)
li = vi+1 - vi

let ti = n.cross(li) / |n.cross(li)|

caclulate dti/dv \in R^{3 * 9}

caclulate d^2ti/dv^2 \in R^{3 * 9 * 9}
*/

typedef Eigen::Matrix<_FLOAT, 9, 9> tMatrix9;
typedef Eigen::Matrix<_FLOAT, 9, 1> tVector9;
typedef Eigen::Matrix<_FLOAT, 3, 9> tMatrix39;
typedef Eigen::Matrix<_FLOAT, 9, 3> tMatrix93;
static tMatrix3 I3 = tMatrix3::Identity();
static tEigenArr<tMatrix39> Ci_array = {};
// calculate t = n.cross(li) / |n.cross(li)|

void CalcCi_array()
{
    if (Ci_array.size() != 0)
        return;
#pragma omp critical
    {
        Ci_array.resize(3, tMatrix39::Zero());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                if (j == i)
                {
                    Ci_array[i].block(0, 3 * j, 3, 3) = -I3;
                }
                else if (j == (i + 1) % 3)
                {
                    Ci_array[i].block(0, 3 * j, 3, 3) = I3;
                }
                else if (j == (i + 2) % 3)
                {
                    Ci_array[i].block(0, 3 * j, 3, 3).setZero();
                }
            }
    }
}
tMatrix39 GetCi(int idx)
{
    CalcCi_array();
    return Ci_array[idx];
}
// tVector3 CalcN(const tVector3 *v_array)
// {
//     tVector3 n = (v_array[1] - v_array[0]).cross(v_array[2] - v_array[1]);
//     return n.normalized();
// }
tVector3 CalcN(const tVector3 *v_array, _FLOAT &length)
{
    tVector3 n = (v_array[1] - v_array[0]).cross(v_array[2] - v_array[1]);
    length = n.norm();
    return n / length;
}

/*
                                                             Calculate dti/dx
                                                         */
tMatrix9 CalcDTDx(const tVector3 *v_array, tMatrix3 *dli_normed_dli,
                  tMatrix39 &D, tMatrix39 &K)
{
    CalcCi_array();
    _FLOAT n_norm = 0;
    tVector3 n_normed = CalcN(v_array, n_norm);
    tVector3 edge_normed[3];
    tVector3 edge_length;
    tMatrix3 edge_skew[3];
    for (int i = 0; i < 3; i++)
    {
        edge_normed[i] = v_array[(i + 1) % 3] - v_array[(i) % 3];
        edge_skew[i] = cMathUtil::VectorToSkewMat(edge_normed[i]);
        K.block(0, 3 * ((i + 2) % 3), 3, 3) = edge_skew[i];
        edge_length[i] = edge_normed[i].norm();
        edge_normed[i] /= edge_length[i];
    }
    D = (tMatrix3::Identity() - n_normed * n_normed.transpose()) / (n_norm)*K;

    // now D = dn/dx, where n is normalized normal vector
    tMatrix3 N_skew = cMathUtil::VectorToSkewMat(n_normed);

    tMatrix9 res = tMatrix9::Zero();
    for (int i = 0; i < 3; i++)
    {
        dli_normed_dli[i] = 1.0 / edge_length[i] *
                            (I3 - edge_normed[i] * edge_normed[i].transpose());
        res.block(3 * i, 0, 3, 9) = -N_skew * dli_normed_dli[i] * Ci_array[i]

                                    + 1.0 / edge_length[i] * edge_skew[i] * D;
    }
    SIM_ASSERT(res.hasNaN() == false);
    return res;
}
void CalcDTDx_times_u_debug(const tVector3 *v_array, const tVector3 &u,
                            tVector9 *debug_Ti0, tVector9 *debug_Ti1)
{
    CalcCi_array();
    _FLOAT n_norm = 0;
    tVector3 n_normed = CalcN(v_array, n_norm);
    tVector3 edge_normed[3];
    tVector3 edge_length;
    tMatrix3 edge_skew[3];
    tMatrix39 D = tMatrix39::Zero();
    for (int i = 0; i < 3; i++)
    {
        edge_normed[i] = v_array[(i + 1) % 3] - v_array[(i) % 3];
        edge_skew[i] = cMathUtil::VectorToSkewMat(edge_normed[i]);
        D.block(0, 3 * ((i + 2) % 3), 3, 3) = edge_skew[i];
        edge_length[i] = edge_normed[i].norm();
        edge_normed[i] /= edge_length[i];
    }
    D = (tMatrix3::Identity() - n_normed * n_normed.transpose()) / (n_norm)*D;

    // now D = dn/dx, where n is normalized normal vector
    tMatrix3 N_skew = cMathUtil::VectorToSkewMat(n_normed);

    tMatrix9 res = tMatrix9::Zero();
    for (int i = 0; i < 3; i++)
    {
        tMatrix3 Gammai = 1.0 / edge_length[i] *
                          (I3 - edge_normed[i] * edge_normed[i].transpose());

        const tMatrix93 Tij_mat_part0 =
            Ci_array[i].transpose() * Gammai * N_skew;
        const tMatrix93 Tij_mat_part1 =
            1.0 / edge_length[i] * D.transpose() * edge_skew[i];

        debug_Ti0[i] = Tij_mat_part0 * u;
        debug_Ti1[i] = Tij_mat_part1 * u;
    }
}

tVector9 CalcTi(const tVector3 *v_array)
{
    // ti = normed(li \times n)
    _FLOAT n_length = 0;
    tVector3 n_normed = CalcN(v_array, n_length);
    tVector9 res = tVector9::Zero();

    for (int i = 0; i < 3; i++)
    {
        tVector3 tmp = (v_array[(i + 1) % 3] - v_array[i]).normalized();
        res.segment(3 * i, 3) = tmp.cross(n_normed);
    }
    return res;
}

tMatrix9 CalcDTDx(const tVector3 *v_array)
{
    tMatrix3 dli_normed_dli[3];
    tMatrix39 D, K;
    return CalcDTDx(v_array, dli_normed_dli, D, K);
}
void VerifyDTDx()
{
    tVector3 v_array[3];
    for (int i = 0; i < 3; i++)
    {
        v_array[i].setRandom();
    }

    // 1.
    tMatrix9 dtidx_ana = CalcDTDx(v_array);
    tMatrix9 dtidx_num = tMatrix9::Zero();
    tVector9 ti_old = CalcTi(v_array);
    _FLOAT eps = 1e-6;
    for (int i = 0; i < 9; i++)
    {
        v_array[i / 3][i % 3] += eps;
        tVector9 ti_new = CalcTi(v_array);
        dtidx_num.col(i) = (ti_new - ti_old) / eps;
        v_array[i / 3][i % 3] -= eps;
    }
    tMatrix9 diff = (dtidx_num - dtidx_ana).cwiseAbs();
    _FLOAT max_diff = diff.maxCoeff();
    std::cout << "dtidx_ana = \n" << dtidx_ana << std::endl;
    std::cout << "dtidx_num = \n" << dtidx_num << std::endl;
    std::cout << "diff = \n" << diff << std::endl;
    std::cout << "max_diff = \n" << max_diff << std::endl;
}