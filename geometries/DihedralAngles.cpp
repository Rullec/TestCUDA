#include "DihedralAngles.h"
#include "geometries/Primitives.h"
#include "utils/DefUtil.h"
#include "utils/LogUtil.h"
#include <algorithm>
#include <iostream>
#include <set>

static double CalcTriAreaFromEdgeLenght(double a, double b, double c)
{
    double p = (a + b + c) / 2;
    double S = std::sqrt(p * (p - a) * (p - b) * (p - c));
    return S;
}
tVectorX GetXPos(const std::vector<tVertexPtr> &mVertexArray)
{
    tVectorX vec = tVectorX::Zero(mVertexArray.size() * 3);
    for (int i = 0; i < mVertexArray.size(); i++)
    {
        vec.segment(3 * i, 3) = mVertexArray[i]->mPos.segment(0, 3);
    }
    return vec;
}
static tVector12 CalcDBetaDx(const tVector3 &v0, const tVector3 &v1,
                      const tVector3 &v2, const tVector3 &v3);
static void VerifyDBetaDx();
static void VerifyDBetaDnormal();
cDihedralAngles::cDihedralAngles() {}

void cDihedralAngles::Init(const std::vector<tVertexPtr> &v_array_,
                           const std::vector<tEdgePtr> &e_array_,
                           const std::vector<tTrianglePtr> &t_array_)
{
    mVertexArray = v_array_;
    mEdgeArray = e_array_;
    mTriangleArray = t_array_;
    mNumOfVertices = v_array_.size();
    mNumOfEdges = e_array_.size();
    mNumOfTriangles = t_array_.size();
    InitConstraintVertex();

    int numOfVertices = v_array_.size();
    int numOfEdges = e_array_.size();
    int numOfTris = t_array_.size();

    // 1. raw edge length
    mRawEdgeLengthArray.resize(numOfEdges);
    mBetaArray.resize(numOfEdges);
    mBetaGradArray.resize(numOfEdges);
    mBetaHessArray.resize(numOfEdges);
    for (int i = 0; i < numOfEdges; i++)
    {
        auto edge = mEdgeArray[i];
        int v0 = edge->mId0;
        int v1 = edge->mId1;
        mRawEdgeLengthArray[i] =
            (mVertexArray[v0]->mPos - mVertexArray[v1]->mPos).norm();
        // std::cout << "raw edge length " << i << " = " <<
        // mRawEdgeLengthArray[i]
        //           << std::endl;
    }

    // 2. raw triangle area
    mRawTriangleAreaArray.resize(numOfTris);
    for (int i = 0; i < numOfTris; i++)
    {
        double e0_length = mRawEdgeLengthArray[mTriangleArray[i]->mEId0];
        double e1_length = mRawEdgeLengthArray[mTriangleArray[i]->mEId1];
        double e2_length = mRawEdgeLengthArray[mTriangleArray[i]->mEId2];
        mRawTriangleAreaArray[i] =
            CalcTriAreaFromEdgeLenght(e0_length, e1_length, e2_length);
        // std::cout << "raw tri area " << i << " = " <<
        // mRawTriangleAreaArray[i]
        //           << std::endl;
    }

    // 3. raw height (for inner triangles)
    mRawHeightArray.resize(numOfEdges);
    tVector2 height = tVector2::Zero();
    for (int i = 0; i < numOfEdges; i++)
    {
        auto e = mEdgeArray[i];
        if (e->mIsBoundary == false)
        {
            // 3.1 get triangle id (left and right)
            int t0 = e->mTriangleId0;
            int t1 = e->mTriangleId1;
            // 3.2 get edge length and triangle area
            double a0 = mRawTriangleAreaArray[t0];
            double a1 = mRawTriangleAreaArray[t1];
            double e_length = mRawEdgeLengthArray[i];
            // 3.3 get height. S = 0.5 * e * h; h = 2 * S / e
            height = 2 * tVector2(a0, a1) / e_length;
            // std::cout << "edge height " << i << " =  " << height.transpose()
            //           << std::endl;
        }
        else
        {
            height.setZero();
        }
        mRawHeightArray[i] = height;
    }

    Update(GetXPos(mVertexArray));
}

/*
    beta = arctan2(y, x) - \pi
    n0 = (x1 - x0) \times (x2 - x1)
    n1 = (x0 - x1) \times (x3 - x0)

    \bar n = n / |n|

    I'_vec = (I3 - bar_v * bar_v^T) / |v|

    d(beta)/dn0 = - sign(sin \beta) / sqrt(1 - x*x) * I'_{n0}^T * \bar n_1
    d(beta)/dn1 = - sign(sin \beta) / sqrt(1 - x*x) * I'_{n1}^T * \bar n_0
*/
static tMatrix3 CalcIprime(const tVector3 &vec)
{
    tMatrix3 res = tMatrix3::Identity();
    return (res - vec.normalized() * vec.normalized().transpose()) / vec.norm();
}

/*
    x = n0_bar.T * n1_bar
    y = e_bar.T * (n0_bar \times n1_bar)
    beta = arctan(y, x) - pi
*/
static  double CalcBeta(const tVector3 &n0_bar, const tVector3 &n1_bar,
                const tVector3 &e_bar)
{
    /*
        x = n0_bar . n1_bar
        y = e_bar . (n0_bar \times n1_bar)
        beta = arctan2(y, x) - pi
    */
    double x = n0_bar.dot(n1_bar);
    double y = e_bar.dot(n0_bar.cross(n1_bar));
    double beta = std::atan2(y, x);
    return beta;
}

static  void CalcNormalAndEdge(const tVector3 &v0, const tVector3 &v1,
                       const tVector3 &v2, const tVector3 &v3, tVector3 &n0,
                       tVector3 &n1, tVector3 &e)
{
    n0 = (v1 - v0).cross(v2 - v1);
    n1 = (v0 - v1).cross(v3 - v0);
    e = (v1 - v0);
    if (n0.hasNaN() == true || n1.hasNaN() == true || e.hasNaN() == true)
    {
        std::cout << "n0, n1, e has Nan\n";
        std::cout << "n0 = " << n0.transpose() << std::endl;
        std::cout << "n1 = " << n1.transpose() << std::endl;
        std::cout << "e = " << e.transpose() << std::endl;
        std::cout << "v0 = " << v0.transpose() << std::endl;
        std::cout << "v1 = " << v1.transpose() << std::endl;
        std::cout << "v2 = " << v2.transpose() << std::endl;
        std::cout << "v3 = " << v3.transpose() << std::endl;
        exit(1);
    }
}

static double CalcBeta(const tVector3 &v0, const tVector3 &v1, const tVector3 &v2,
                const tVector3 &v3)
{
    tVector3 n0, n1, e;
    CalcNormalAndEdge(v0, v1, v2, v3, n0, n1, e);
    return CalcBeta(n0.normalized(), n1.normalized(), e.normalized());
}

static void CalcDBetaDnormal_and_De(const tVector3 &n0, const tVector3 &n1,
                             const tVector3 &e, tVector3 &dbeta_dn0,
                             tVector3 &dbeta_dn1, tVector3 &dbeta_de)
{
    // 1. prepare
    tVector3 n0_bar = n0.normalized();
    tVector3 n1_bar = n1.normalized();
    tVector3 e_bar = e.normalized();
    double x = n0_bar.dot(n1_bar);
    double y = e_bar.dot(n0_bar.cross(n1_bar));
    double beta = CalcBeta(n0_bar, n1_bar, e_bar);
    // std::cout << "beta = " << beta << std::endl;
    double sign_sinb = std::sin(beta) > 0 ? 1.0 : -1.0;
    double sign_cosb = std::cos(beta) > 0 ? 1.0 : -1.0;
    // 2. I'
    tMatrix3 Iprime0 = CalcIprime(n0), Iprime1 = CalcIprime(n1);
    double deno = std::sqrt(SIM_MAX(1 - x * x, 0));
    if (deno < 1e-10)
    {
        deno = 1e-10;
    }
    if (std::isnan(deno))
    {
        std::cout << "deno is nan, deno = " << deno << std::endl;
        std::cout << "x = " << x << std::endl;
        exit(1);
    }
    double factor = -sign_sinb * 1.0 / deno;
    if (std::isnan(factor))
    {
        std::cout << "factor is nan, factor = " << factor << std::endl;
        exit(1);
    }
    dbeta_dn0 = factor * Iprime0.transpose() * n1.normalized();
    dbeta_dn1 = factor * Iprime1.transpose() * n0.normalized();
    dbeta_de = sign_cosb / std::sqrt(1 - y * y) * CalcIprime(e).transpose() *
               (n0_bar.cross(n1_bar));
    if (dbeta_dn0.hasNaN() == true)
    {
        std::cout << "Iprime0 = \n" << Iprime0 << std::endl;
        std::cout << "dbeta_dn0 hasNan = " << dbeta_dn0.transpose()
                  << std::endl;
        std::cout << "factor = " << factor << std::endl;
        std::cout << "Iprime0 = " << Iprime0 << std::endl;
        std::cout << "n1 normalized = " << n1.normalized().transpose()
                  << std::endl;
        std::cout << "n1 " << n1.transpose() << std::endl;
        exit(1);
    }
    if (dbeta_dn1.hasNaN() == true)
    {
        std::cout << "dbeta_dn1 hasNan = " << dbeta_dn1.transpose()
                  << std::endl;
        exit(1);
    }
    if (dbeta_de.hasNaN() == true)
    {
        std::cout << "dbeta_de hasNan = " << dbeta_de.transpose() << std::endl;
        exit(1);
    }
}

static tVector12 CalcDBetaDx(const tVector3 &v0, const tVector3 &v1,
                      const tVector3 &v2, const tVector3 &v3)
{
    // 1. calculate beta
    tVector3 n0, n1, e;
    CalcNormalAndEdge(v0, v1, v2, v3, n0, n1, e);
    tVector3 n0_bar = n0.normalized(), n1_bar = n1.normalized(),
             e_bar = e.normalized();
    double beta = CalcBeta(n0_bar, n1_bar, e_bar);

    // 2. calculate dbeta/dn0, dbeta/dn1
    tVector3 dbeta_dn0, dbeta_dn1, dbeta_de;
    CalcDBetaDnormal_and_De(n0, n1, e, dbeta_dn0, dbeta_dn1, dbeta_de);
    // std::cout << "dbeta_dn0 = " << dbeta_dn0.transpose() << std::endl;
    // std::cout << "dbeta_dn1 = " << dbeta_dn1.transpose() << std::endl;
    /*
        3. calcualte
            dbeta / dv0 =
                (v1 - v2) \times dbdn0
                +
                (v1 - v3) \times dbdn1

            dbeta / dv1 =
                (v2 - v0) \times dbdn0
                +
                (v0 - v3) \times dbdn1

            dbeta / dv2 =
                (v0 - v1) \times dbdn0

            dbeta / dv3 =
                (v1 - v0) \times dbdn1
    */
    // std::cout << "dbeta_de = " << dbeta_de.transpose() << std::endl;
    tVector3 dbeta_dv0 =
        (v1 - v2).cross(dbeta_dn0) + (v3 - v1).cross(dbeta_dn1) - dbeta_de;
    tVector3 dbeta_dv1 =
        (v2 - v0).cross(dbeta_dn0) + (v0 - v3).cross(dbeta_dn1) + dbeta_de;
    tVector3 dbeta_dv2 = (v0 - v1).cross(dbeta_dn0);
    tVector3 dbeta_dv3 = (v1 - v0).cross(dbeta_dn1);
    tVector12 res(12);
    res.segment(0, 3) = dbeta_dv0;
    res.segment(3, 3) = dbeta_dv1;
    res.segment(6, 3) = dbeta_dv2;
    res.segment(9, 3) = dbeta_dv3;
    return res;
}
static void VerifyDBetaDnormal()
{
    for (int i = 0; i < 10; i++)
    {
        tVector3 n0 = tVector3::Random(), n1 = tVector3::Random();
        tVector3 e = n0.cross(n1);

        // tVector3 n0 = tVector3(-0.11283083, -0.00337882, -0.03312034);
        // tVector3 n1 = tVector3(0.17173063, 0.06385464, -0.03608133);
        // tVector3 e = tVector3(-0.10729597, 0.4681182, 0.31776865);

        // 1. calculate old beta
        double old_beta =
            CalcBeta(n0.normalized(), n1.normalized(), e.normalized());

        // 2. calc ana deriv
        tVector3 dbeta_dn0_ana = tVector3::Zero();
        tVector3 dbeta_dn1_ana = tVector3::Zero();
        tVector3 _dbeta_de = tVector3::Zero();
        CalcDBetaDnormal_and_De(n0, n1, e, dbeta_dn0_ana, dbeta_dn1_ana,
                                _dbeta_de);

        // 3. calc num deriv
        double eps = 1e-5;
        tVector3 dbeta_dn0_num = tVector3::Zero();
        tVector3 dbeta_dn1_num = tVector3::Zero();
        for (int i = 0; i < 3; i++)
        {
            n0[i] += eps;
            double new_beta =
                CalcBeta(n0.normalized(), n1.normalized(), e.normalized());
            dbeta_dn0_num[i] = (new_beta - old_beta) / eps;
            n0[i] -= eps;
        }
        for (int i = 0; i < 3; i++)
        {
            n1[i] += eps;
            double new_beta =
                CalcBeta(n0.normalized(), n1.normalized(), e.normalized());
            dbeta_dn1_num[i] = (new_beta - old_beta) / eps;
            n1[i] -= eps;
        }
        tVector3 diff_n0 = (dbeta_dn0_ana - dbeta_dn0_num);
        tVector3 diff_n1 = (dbeta_dn1_ana - dbeta_dn1_num);
        double diff_n0_norm = diff_n0.norm();
        double diff_n1_norm = diff_n1.norm();
        if (diff_n0_norm > 0.1 || diff_n1_norm > 0.1)
        {
            std::cout << "dbeta_dn0_ana = " << dbeta_dn0_ana.transpose()
                      << std::endl;
            std::cout << "dbeta_dn0_num = " << dbeta_dn0_num.transpose()
                      << std::endl;
            std::cout << "dbeta_dn0_diff = " << diff_n0.transpose()
                      << " norm = " << diff_n0_norm << std::endl;
            std::cout << "dbeta_dn1_ana = " << dbeta_dn1_ana.transpose()
                      << std::endl;
            std::cout << "dbeta_dn1_num = " << dbeta_dn1_num.transpose()
                      << std::endl;
            std::cout << "dbeta_dn1_diff = " << diff_n1.transpose()
                      << " norm = " << diff_n1_norm << std::endl;
            exit(1);
        }
    }
    std::cout << "VerifyDBetaDnormal succ\n";
}
static void VerifyDBetaDx()
{
    tVectorX pos = tVectorX::Random(12);
    // pos.segment(0, 3) = tVector3(0.5488135, 0.71518937, 0.60276338);
    // pos.segment(3, 3) = tVector3(0.54488318, 0.4236548, 0.64589411);
    // pos.segment(6, 3) = tVector3(0.43758721, 0.891773, 0.96366276);
    // pos.segment(9, 3) = tVector3(0.38344152, 0.79172504, 0.52889492);

    // pos.segment(0, 3) = tVector3(0.54488318, 0.4236548, 0.64589411);
    // pos.segment(3, 3) = tVector3(0.43758721, 0.891773, 0.96366276);
    // pos.segment(6, 3) = tVector3(0.5488135, 0.71518937, 0.60276338);
    // pos.segment(9, 3) = tVector3(0.38344152, 0.79172504, 0.52889492);

    pos << -0.05, 0.39995, -0.05, 0.05, 0.40004, 0.05, 0.05, 0.39996, -0.05,
        -0.05, 0.40004, 0.05;
    // pos[0] =
    // 1. get old value
    tVectorX v0 = pos.segment(0, 3), v1 = pos.segment(3, 3),
             v2 = pos.segment(6, 3), v3 = pos.segment(9, 3);
    double old_beta = CalcBeta(v0, v1, v2, v3);
    std::cout << "old_beta = " << old_beta << std::endl;
    // 2. get ana deriv
    tVectorX dbeta_dx_ana = CalcDBetaDx(v0, v1, v2, v3);
    tVectorX dbeta_dx_num = tVectorX::Zero(12);

    double eps = 1e-8;
    for (int i = 0; i < 12; i++)
    {
        pos[i] += eps;
        v0 = pos.segment(0, 3);
        v1 = pos.segment(3, 3);
        v2 = pos.segment(6, 3);
        v3 = pos.segment(9, 3);
        double new_beta = CalcBeta(v0, v1, v2, v3);
        dbeta_dx_num[i] = (new_beta - old_beta) / eps;
        pos[i] -= eps;
    }
    // 3. get num deriv
    tVectorX diff = dbeta_dx_ana - dbeta_dx_num;
    std::cout << "dbeta_dx_ana = " << dbeta_dx_ana.transpose() << std::endl;
    std::cout << "dbeta_dx_num = " << dbeta_dx_num.transpose() << std::endl;
    std::cout << "diff = " << diff.transpose() << std::endl;
}
#include "geometries/DihedralAnglesMachine.h"
#include "utils/ProfUtil.h"
using namespace nDihedralAnglesMachine;
void cDihedralAngles::Update(const tVectorX &xcur, bool update_grad,
                             bool update_hess)
{
    // printf("dihedral materia update now!\n");

    int numOfEdges = this->mEdgeArray.size();

    // OMP_PARALLEL_FOR(OMP_MAX_THREADS)
    for (int i = 0; i < numOfEdges; i++)
    {
        auto cur_e = mEdgeArray[i];
        if (cur_e->mIsBoundary)
            continue;
        // printf("-------------[dih] edge %d ------------\n", i);
        // 1. calculate force for each element, storage into the vector

        /*
        all variables are static, except dbeta (Gauss-Newton method, quasi
        Newton)

        shape   = e^2 / (2 * (A1 + A2))
                = e / (h1 + h2)
        dt0 = -1.0 / h1 * n0;
        dt1 = cos(alpha_2) / h1 * n0 + cos(alpha_2_tile) / h1 * n1
        dt2 = cos(alpha_1) / h2 * n0 + cos(alpha_1_tile) / h2 * n1

        dbeta = [dt0, dt1, dt2, dt3] \in R^{12}

        f = -modulus * shape * beta  * dbeta / 2
        H = -modulus * shape * dbeta * dbeta.T
        */
        tVector4i v_id_lst = mConstraintVertexLst[i];
        tVector3 v0 = xcur.segment(3 * v_id_lst[0], 3);
        tVector3 v1 = xcur.segment(3 * v_id_lst[1], 3);
        tVector3 v2 = xcur.segment(3 * v_id_lst[2], 3);
        tVector3 v3 = xcur.segment(3 * v_id_lst[3], 3);

        auto FourVec3ToVec12 = [v0, v1, v2, v3]() -> tVector12
        {
            tVector12 out;
            out.segment<3>(0) = v0;
            out.segment<3>(3) = v1;
            out.segment<3>(6) = v2;
            out.segment<3>(9) = v3;
            return out;
        };

        tVector12 vec12 = FourVec3ToVec12();
        // ===================== self calculation =====================
        // mBetaArray[i] = CalcBeta(v0, v1, v2, v3);

        // if (update_grad)
        //     mBetaGradArray[i] = CalcDBetaDx(v0, v1, v2, v3);

        // ==================== machine calculation ==================
        //  = nDihedralAnglesMachine::Calcd(v0, v1, v2, v3);
        mBetaArray[i] = nDihedralAnglesMachine::CalcTheta(vec12);
        if (update_grad)
            mBetaGradArray[i] = nDihedralAnglesMachine::CalcdThetadx(vec12);

        // ==================== end ==================
        if (update_hess)
        {
            mBetaHessArray[i].noalias() =
                nDihedralAnglesMachine::CalcdTheta2dx2(vec12);
        }

        // if (mBetaGradArray[i].hasNaN() == true)
        // {
        //     SIM_ERROR("[error] dihedral grad for edge {} has nan = {}, v0 {}
        //     "
        //               "v1 {} v2 {} v3 {}",
        //               i, mBetaGradArray[i].transpose(), v0.transpose(),
        //               v1.transpose(), v2.transpose(), v3.transpose());
        //     exit(1);
        // }
        // // 2. calcualte shape
        // double e = mRawEdgeLengthArray[i];
        // tVector2 height = mRawHeightArray[i];
        // double height_sum = SIM_MAX(1e-7f, height[0] + height[1]);

        // tMatrix12 hessian = dbeta_dx * dbeta_dx.transpose();
        // if (hessian.hasNaN() == true)
        // {
        //     std::cout << "edge " << i << " hessian nan = \n"
        //               << hessian << std::endl;
        //     std::cout << "dAdx = \n" << dbeta_dx << std::endl;
        //     exit(1);
        // }
        // // std::cout << "force = " << force.transpose() << std::endl;
        // // std::cout << "hessian = " << hessian.transpose() << std::endl;
        // mBetaHessArray[i].setZero();

        // // assign the force and hessian into the global stiffness matrix
        // mEleGradArray[i] = force;
        // // exit(1);
    }
    // printf("machine %.3e, self %.3e\n", cProfUtil::GetElapsedTime("machine"),
    //        cProfUtil::GetElapsedTime("self"));
}
void cDihedralAngles::CheckHess()
{

    tVectorX xcur = GetXPos(mVertexArray);
    Update(xcur, true, true);
    for (int eid = 0; eid < mNumOfEdges; eid++)
    {
        if (mEdgeArray[eid]->mIsBoundary)
            continue;

        tVector4i vids = mConstraintVertexLst[eid];
        _FLOAT old_theta = GetAngles()[eid];
        tVector12 old_grad = GetGradAngles()[eid];
        if (std::fabs(old_theta) < 1e-2)
        {
            printf("[check] dihedral edge %d angle %.1e is too small, too much "
                   "error in num deriv, ignore\n",
                   eid, old_theta);
            continue;
        }
        tMatrix12 hess_ana = GetHessAngles()[eid];
        tMatrix12 hess_num = tMatrix12::Zero();
        _FLOAT eps = 1e-7;
        for (int i = 0; i < 3 * vids.size(); i++)
        {
            xcur[3 * vids[i / 3] + i % 3] += eps;
            Update(xcur, true, false);

            tVector12 new_grad = this->GetGradAngles()[eid];
            hess_num.col(i) = (new_grad - old_grad) / eps;
            xcur[3 * vids[i / 3] + i % 3] -= eps;
        }
        Update(xcur, true, false);
        tMatrix12 diff = (hess_ana - hess_num).cwiseAbs();
        _FLOAT max_diff = diff.maxCoeff();
        if (max_diff > 1e-1)
        {

            SIM_WARN("check hess for dihedral angles edge {}, failed, angle = "
                     "{}\n",
                     eid, mBetaArray[eid]);
            for (int i = 0; i < 4; i++)
            {
                std::cout << "v" << i << " = "
                          << xcur.segment(3 * vids[i], 3).transpose()
                          << std::endl;
            }
            std::cout << "hess_ana = \n" << hess_ana << std::endl;
            std::cout << "hess_num = \n" << hess_num << std::endl;
            std::cout << "diff = \n" << diff << std::endl;
            std::cout << "max_diff = \n" << max_diff << std::endl;
            // exit(1);
        }
        printf("check dthetadx succ for e %d\n", eid);
    }
    Update(xcur, true, true);
}
void cDihedralAngles::CheckGrad()
{
    tVectorX xcur = GetXPos(mVertexArray);
    Update(xcur, true, false);
    for (int eid = 0; eid < mNumOfEdges; eid++)
    {
        if (mEdgeArray[eid]->mIsBoundary)
            continue;

        tVector4i vids = mConstraintVertexLst[eid];
        _FLOAT old_theta = this->GetAngles()[eid];
        if (std::fabs(old_theta) < 1e-2)
        {
            printf("[check] dihedral edge %d angle %.1e is too small, too much "
                   "error in num deriv, ignore\n",
                   eid, old_theta);
            continue;
        }
        tVector12 dtdx_ana = GetGradAngles()[eid];
        tVector12 dtdx_num = tVector12::Zero();
        _FLOAT eps = 1e-8;
        for (int i = 0; i < 3 * vids.size(); i++)
        {
            xcur[3 * vids[i / 3] + i % 3] += eps;
            Update(xcur, false, false);

            _FLOAT new_theta = this->GetAngles()[eid];
            dtdx_num[i] = (new_theta - old_theta) / eps;
            xcur[3 * vids[i / 3] + i % 3] -= eps;
        }
        Update(xcur, false, false);
        tVector12 diff = (dtdx_ana - dtdx_num).cwiseAbs();
        _FLOAT max_diff = diff.maxCoeff();
        if (max_diff > 1e-1)
        {

            SIM_WARN("check grad for dihedral angles edge {}, failed, angle = "
                     "{}\n",
                     eid, mBetaArray[eid]);
            for (int i = 0; i < 4; i++)
            {
                std::cout << "v" << i << " = "
                          << xcur.segment(3 * vids[i], 3).transpose()
                          << std::endl;
            }
            std::cout << "dtdx_ana = " << dtdx_ana.transpose() << std::endl;
            std::cout << "dtdx_num = " << dtdx_num.transpose() << std::endl;
            std::cout << "diff = " << diff.transpose() << std::endl;
            std::cout << "max_diff = " << max_diff << std::endl;
            // exit(1);
        }
        printf("check dthetadx succ for e %d\n", eid);
    }
    Update(xcur, true, false);
}
void cDihedralAngles::InitConstraintVertex()
{
    mConstraintVertexLst.clear();

    for (auto &e : mEdgeArray)
    {
        if (e->mIsBoundary == true)
        {
            mConstraintVertexLst.push_back(tVector4i(-1, -1, -1, -1));
        }
        else
        {
            int v0 = e->mId0, v1 = e->mId1;
            int v2 = mTriangleArray[e->mTriangleId0]->SelectAnotherVertex(v0,
                                                                          v1),
                v3 = mTriangleArray[e->mTriangleId1]->SelectAnotherVertex(v0,
                                                                          v1);
            int v_lst[4] = {v0, v1, v2, v3};
            mConstraintVertexLst.push_back(tVector4i(v0, v1, v2, v3));
        }
    }
}

const std::vector<_FLOAT> cDihedralAngles::GetAngles() const
{
    return this->mBetaArray;
}
const tEigenArr<tVector12> cDihedralAngles::GetGradAngles() const
{
    return this->mBetaGradArray;
}
const tEigenArr<tMatrix12> cDihedralAngles::GetHessAngles() const
{
    return this->mBetaHessArray;
}

const std::vector<_FLOAT> &cDihedralAngles::GetRawTriangleAreaArray() const
{
    return this->mRawTriangleAreaArray;
}

const tVector4i cDihedralAngles::GetStencil(int eid) const
{
    return this->mConstraintVertexLst[eid];
}