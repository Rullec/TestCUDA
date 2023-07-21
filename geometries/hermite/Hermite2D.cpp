#include "utils/DefUtil.h"
#include "utils/EigenUtil.h"
#include "utils/LogUtil.h"
#include "utils/TimeUtil.hpp"
#include <iostream>
#include <vector>

tVector4 Vandermonde(const _FLOAT &val)
{
    tVector4 ret = tVector4::Ones();

    for (int i = 1; i < 4; i++)
        ret[i] = ret[i - 1] * val;
    return ret;
}

tVector4 dVandermonde_dx(const _FLOAT &val)
{
    tVector4 ret = tVector4::Ones();
    ret << 0, 1, 2 * val, 3 * val * val;
    return ret;
}

void EvalBicubicHermite(const tVector4 &fval_lst, const tVector4 &fx_lst,
                        const tVector4 &fy_lst, const tVector4 &fxy_lst,
                        const tVector2 &x_range, const tVector2 &y_range,
                        _FLOAT X_, _FLOAT Y_, _FLOAT &f, _FLOAT &dfdx,
                        _FLOAT &dfdy)
{
    // assert len(fval_lst) == 4
    // assert len(fx_lst) == 4
    // assert len(fy_lst) == 4
    // assert len(fxy_lst) == 4
    // assert len(x_range) == 2 and len(y_range) == 2
    // assert isnumber(X) and isnumber(Y)

    // tMatrix4 Hm = np.array([[1 , 0 , 0 , 0],
    //     [0 , 0 , 1 , 0] ,
    //     [-3 , 3 , -2 , -1 ],
    //     [2 , -2 , 1 , 1 ]
    //     ])
    tMatrix4 Hm;
    Hm << 1, 0, 0, 0, 0, 0, 1, 0, -3, 3, -2, -1, 2, -2, 1, 1;
    // std::cout << "Hm = \n" << Hm << std::endl;
    // scale x, y to [0, 1]
    _FLOAT x0 = x_range[0], x1 = x_range[1];
    _FLOAT y0 = y_range[0], y1 = y_range[1];

    _FLOAT dx = x1 - x0;
    _FLOAT dy = y1 - y0;
    _FLOAT X = (X_ - x0) / dx;
    _FLOAT Y = (Y_ - y0) / dy;

    tVector4 X_orders = Vandermonde(X);
    tVector4 Y_orders = Vandermonde(Y);

    tVector4 dXdx = dVandermonde_dx(X) / dx;
    tVector4 dYdy = dVandermonde_dx(Y) / dy;
    // X_orders = np.vander([X], N = 4, increasing=True)
    // Y_orders = np.vander([Y], N = 4, increasing=True).T

    tMatrix4 P = tMatrix4::Zero();
    auto set_block =
        [](tMatrix4 &P, int st_row, int st_col, const tVector4 &val_lst)
    {
        P(st_row, st_col) = val_lst[0];
        P(st_row, st_col + 1) = val_lst[1];
        P(st_row + 1, st_col) = val_lst[2];
        P(st_row + 1, st_col + 1) = val_lst[3];
    };

    set_block(P, 0, 0, fval_lst);
    set_block(P, 2, 0, fx_lst);
    set_block(P, 0, 2, fy_lst);
    set_block(P, 2, 2, fxy_lst);

    tMatrix4 dxdy = tMatrix4::Zero();
    dxdy << 1, 1, dy, dy, 1, 1, dy, dy, dx, dx, dx * dy, dx * dy, dx, dx,
        dx * dy, dx * dy;
    // std::cout << "dxdy = \n" << dxdy << std::endl;
    // std::cout << "before P = \n" << P << std::endl;
    P = P.cwiseProduct(dxdy);
    // std::cout << "P = \n" << P << std::endl;

    // U * Hm * P * Hm.T * Y_orders
    tMatrix4 tmp = Hm * P * Hm.transpose();
    // std::cout << "tmp = \n" << tmp << std::endl;
    // std::cout << "X_orders = " << X_orders.transpose() << std::endl;
    // std::cout << "Y_orders = " << Y_orders.transpose() << std::endl;
    f = X_orders.dot(tmp * Y_orders);

    dfdx = dXdx.dot(tmp * Y_orders);
    dfdy = X_orders.dot(tmp * dYdy);
    // tmp = np.einsum('ij,jk,lk->il', Hm, P, Hm);
    // P_val = np.einsum("ij,jk,kl->il", X_orders, tmp, Y_orders);

    // return P_val;
}

void EvalPiecewiseBicubicHermte(const std::vector<_FLOAT> &x_lst,
                                const std::vector<_FLOAT> &y_lst,
                                const std::vector<_FLOAT> &fval_lst,
                                const std::vector<_FLOAT> &fx_lst,
                                const std::vector<_FLOAT> &fy_lst,
                                const std::vector<_FLOAT> &fxy_lst,
                                _FLOAT x_sample, _FLOAT y_sample,

                                _FLOAT &func, _FLOAT &dfdx, _FLOAT &dfdy)
{
    /*
    1. Give the piecewise bicubic surface
        x_lst, y_lst, fval_lst, fx_lst, fy_lst, fxy_lst = surf
        x_lst: length N, the x coordinates of grid points
        y_lst: length M, the y coordinates of grid points

        for point (i, j). i \in [0,M-1], j \in [0, N-1]
        we store it in row-major

        val(i, j) = val[i * N + j]

        fval_lst: length N*M
        fx_lst: length N*M
        fy_lst: length N*M
        fxy_lst: length N*M
    2. Give the Q sample pts
        x_sample_lst: length Q
        y_sample_lst: length Q
    */
    int N = x_lst.size();
    int M = y_lst.size();
    SIM_ASSERT(fval_lst.size() == N * M);
    SIM_ASSERT(fx_lst.size() == N * M);
    SIM_ASSERT(fy_lst.size() == N * M);
    SIM_ASSERT(fxy_lst.size() == N * M);

    // SIM_ASSERT(x_sample_lst.size() == y_sample_lst.size());

    auto get_val_from_lb_corner =
        [N](int i, int j, const std::vector<_FLOAT> &val_lst) -> tVector4
    {
        return tVector4(val_lst[i * N + j], val_lst[(i + 1) * N + j],
                        val_lst[i * N + j + 1], val_lst[(i + 1) * N + j + 1]);
    };

    // func_lst.clear();
    // dfdx_lst.clear();
    // dfdy_lst.clear();

    // for (int idx = 0; idx < x_sample_lst.size(); idx++)
    // {

    // _FLOAT
    // cur_x = x_sample_lst[idx], cur_y = y_sample_lst[idx];
    _FLOAT
    cur_x = x_sample, cur_y = y_sample;
    int j = std::lower_bound(x_lst.begin(), x_lst.end(), cur_x) - x_lst.begin();
    int i = std::lower_bound(y_lst.begin(), y_lst.end(), cur_y) - y_lst.begin();
    i = SIM_MAX(SIM_MIN(y_lst.size() - 1, i), 1);
    j = SIM_MAX(SIM_MIN(x_lst.size() - 1, j), 1);
    i = i - 1;
    j = j - 1;
    // printf("[2d_hermite] x %.3f y %.3f, i %d j %d\n", cur_x, cur_y, i,
    // j); for idx, (cur_x, cur_y) in enumerate(zip(x_sample_lst,
    // y_sample_lst)): j = np.searchsorted(x_lst, cur_x, side = 'right') i =
    // np.searchsorted(y_lst, cur_y, side = 'right')

    tVector2 x_range = tVector2(x_lst[j], x_lst[j + 1]);
    tVector2 y_range = tVector2(y_lst[i], y_lst[i + 1]);
    // std::cout << "xrange = " << x_range.transpose()
    //           << " yrange = " << y_range.transpose() << std::endl;
    // 00(BL), 10(BR), 01, 11
    tVector4 piece_fval = get_val_from_lb_corner(i, j, fval_lst);
    tVector4 piece_fx = get_val_from_lb_corner(i, j, fx_lst);
    tVector4 piece_fy = get_val_from_lb_corner(i, j, fy_lst);
    tVector4 piece_fxy = get_val_from_lb_corner(i, j, fxy_lst);

    _FLOAT val = 0, dfdx_tmp = 0, dfdy_tmp = 0;
    EvalBicubicHermite(piece_fval, piece_fx, piece_fy, piece_fxy, x_range,
                       y_range, cur_x, cur_y, val, dfdx_tmp, dfdy_tmp);
    // std::cout << "val = " << val << std::endl;
    func = val;
    dfdx = dfdx_tmp;
    dfdy = dfdy_tmp;
    // }
}

void EvalPiecewiseBicubicHermiteGroup(
    const std::vector<_FLOAT> &x_lst, const std::vector<_FLOAT> &y_lst,
    const std::vector<_FLOAT> &fval_lst, const std::vector<_FLOAT> &fx_lst,
    const std::vector<_FLOAT> &fy_lst, const std::vector<_FLOAT> &fxy_lst,
    std::vector<_FLOAT> x_sample_array, std::vector<_FLOAT> y_sample_array,
    std::vector<_FLOAT> &func_array, std::vector<_FLOAT> &dfdx_array,
    std::vector<_FLOAT> &dfdy_array)
{
    int N = x_lst.size();
    int M = y_lst.size();
    SIM_ASSERT(fval_lst.size() == N * M);
    SIM_ASSERT(fx_lst.size() == N * M);
    SIM_ASSERT(fy_lst.size() == N * M);
    SIM_ASSERT(fxy_lst.size() == N * M);

    // SIM_ASSERT(x_sample_lst.size() == y_sample_lst.size());

    auto get_val_from_lb_corner =
        [N](int i, int j, const std::vector<_FLOAT> &val_lst) -> tVector4
    {
        return tVector4(val_lst[i * N + j], val_lst[(i + 1) * N + j],
                        val_lst[i * N + j + 1], val_lst[(i + 1) * N + j + 1]);
    };

    int j = std::lower_bound(x_lst.begin(), x_lst.end(), x_sample_array[0]) -
            x_lst.begin();
    int i = std::lower_bound(y_lst.begin(), y_lst.end(), y_sample_array[0]) -
            y_lst.begin();
    i = SIM_MAX(SIM_MIN(y_lst.size() - 1, i), 1);
    j = SIM_MAX(SIM_MIN(x_lst.size() - 1, j), 1);
    i = i - 1;
    j = j - 1;

    tVector2 x_range = tVector2(x_lst[j], x_lst[j + 1]);
    tVector2 y_range = tVector2(y_lst[i], y_lst[i + 1]);
    // std::cout << "xrange = " << x_range.transpose()
    //           << " yrange = " << y_range.transpose() << std::endl;
    // 00(BL), 10(BR), 01, 11
    tVector4 piece_fval = get_val_from_lb_corner(i, j, fval_lst);
    tVector4 piece_fx = get_val_from_lb_corner(i, j, fx_lst);
    tVector4 piece_fy = get_val_from_lb_corner(i, j, fy_lst);
    tVector4 piece_fxy = get_val_from_lb_corner(i, j, fxy_lst);

    int num = x_sample_array.size();

    func_array.resize(num);
    dfdx_array.resize(num);
    dfdy_array.resize(num);
    for (int i = 0; i < num; i++)
    {
        EvalBicubicHermite(piece_fval, piece_fx, piece_fy, piece_fxy, x_range,
                           y_range, x_sample_array[i], y_sample_array[i],
                           func_array[i], dfdx_array[i], dfdy_array[i]);
    }
}