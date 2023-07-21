#ifndef DIHEDRAL_ANGLE_MACHINE_H_
#include "utils/BaseTypeUtil.h"
#include "utils/EigenUtil.h"
#include <cassert>
#include <cmath>

namespace nDihedralAnglesMachine
{
typedef _FLOAT Real;
inline Real Power(Real A, Real B) { return std::pow(A, B); }
// inline Real Power(const Real &A, const int &B) { return std::pow(A, B); }
inline Real Power(Real A, int B)
{
    assert(abs(B) < 8 && "require higher pow, implement using std.");
    Real out = 1.0;
    if (B >= 0)
    {
        for (int i = 0; i < B; ++i)
            out *= A;
    }
    else
    {
        Real invA = 1.0 / A;
        for (int i = 0; i < -B; ++i)
            out *= invA;
    }
    return out;
}

inline Real Abs(Real A) { return std::abs(A); }
inline Real Sqrt(Real A) { return std::sqrt(A); }
inline Real Exp(Real A) { return std::exp(A); }
inline Real Sin(Real A) { return std::sin(A); }
inline Real Cos(Real A) { return std::cos(A); }
inline Real Log(Real A) { return std::log(A); }
inline Real ArcCos(Real A) { return std::acos(A); }
inline Real ArcTan(Real X, const Real Y) { return std::atan2(Y, X); }
inline Real Tan(Real X) { return std::tan(X); }
inline Real Sec(Real X) { return 1.0 / std::cos(X); }
inline Real RealAbs(Real X) { return std::abs(X); }
inline Real Sign(Real X) { return (X > 0) - (X < 0); }

typedef Eigen::Matrix<_FLOAT, 12, 1> tVector12;
typedef Eigen::Matrix<_FLOAT, 12, 12> tMatrix12;
_FLOAT CalcTheta(const tVector12 &xcur);
tVector12 CalcdThetadx(const tVector12 &xcur);
tMatrix12 CalcdTheta2dx2(const tVector12 &xcur);
}; // namespace nDihedralAnglesMachine
#endif