// #pragma once
// #include "utils/DefUtil.h"
// #include "utils/EigenUtil.h"
// #include <vector>

// SIM_DECLARE_STRUCT_AND_PTR(tVertex);
// SIM_DECLARE_STRUCT_AND_PTR(tEdge);
// SIM_DECLARE_STRUCT_AND_PTR(tTriangle);
// SIM_DECLARE_CLASS_AND_PTR(cDihedralAngles);
// typedef Eigen::Matrix<int, 6, 1> tVector6i;
// typedef Eigen::Matrix<_FLOAT, 6, 1> tVector6;
// typedef Eigen::Matrix<_FLOAT, 9, 1> tVector9;
// typedef Eigen::Matrix<_FLOAT, 3, 2> tMatrix32;
// typedef Eigen::Matrix<_FLOAT, 3, 9> tMatrix39;
// typedef Eigen::Matrix<_FLOAT, 9, 9> tMatrix9;
// typedef Eigen::Matrix<_FLOAT, 6, 9> tMatrix69;
// typedef Eigen::Matrix<_FLOAT, 3, 18> tMatrix318;
// typedef Eigen::Matrix<_FLOAT, 18, 18> tMatrix18;
// /*
// Calculate I and II fundamental forms

// I = [E & F \\ F & G]
// II = [L & M \\ M & N]
// */
// class cFForms
// {
// public:
//     cFForms();
//     virtual void Init(const std::vector<tVertexPtr> &v_array,
//                       const std::vector<tEdgePtr> &e_array,
//                       const std::vector<tTrianglePtr> &t_array);
//     virtual void Update(const tVectorX &xcur, bool update_grad = true,
//                         bool update_hess = true);
//     virtual const std::vector<tVector3> &GetI() const;
//     virtual const std::vector<tVector3> &GetII() const;
//     virtual const tEigenArr<tMatrix39> &GetIGradPerTriangle() const;
//     virtual const tEigenArr<tMatrix318> &GetIIGradPerTriangle() const;
//     virtual const tEigenArr<tMatrix9> &GetIHess(int idx) const;
//     virtual const tEigenArr<tMatrix18> &GetIIHess(int idx) const;
//     virtual tEigenArr<tMatrix9> GetIHessPerTri(int tid) const;
//     virtual tEigenArr<tMatrix18> GetIIHessPerTri(int tid) const;
//     virtual void CheckFirstFormGrad();
//     virtual void CheckFirstFormHess();
//     virtual void CheckSecondFormGrad();
//     virtual void CheckSecondFormTi0uGrad(); // for hess validation
//     virtual void CheckSecondFormTij0Grad(); // for hess validation
//     virtual void CheckSecondFormGammaiUGrad();
//     virtual void CheckSecondFormCijHess();
//     virtual const tEigenArr<tVector6i> &GetStencil() const;
//     virtual const std::vector<_FLOAT> &GetTriangleAreaStatic() const;
//     virtual cDihedralAnglesPtr GetDihedralAngles() const
//     {
//         return mDihedralAngles;
//     }

// protected:
//     // v, e and t
//     std::vector<tVertexPtr> mVertexArray;
//     std::vector<tEdgePtr> mEdgeArray;
//     std::vector<tTrianglePtr> mTrianglearray;
//     int mNumOfE, mNumOfV, mNumOfT;

//     // static vars
//     tEigenArr<tMatrix2>
//         mDminv; // mDminv = [u1-u0, u2-u0]^-1, used for deformation gradient
//     std::vector<_FLOAT> mTriangleAreaStatic;
//     std::vector<_FLOAT> mBoundaryTriangle;
//     std::vector<_FLOAT> mEdgeLengthStatic;
//     // forms
//     std::vector<tVector3> mFirstArray, mSecondArray;
//     cDihedralAnglesPtr mDihedralAngles;

//     tEigenArr<tMatrix39> mFirstGradArray; // I11, I12, I22
//     tEigenArr<tMatrix318> mSecondGradArray;

//     tEigenArr<tMatrix9> mFirstHessArray[3];   // I11, I12, I22
//     tEigenArr<tMatrix18> mSecondHessArray[3]; // I11, I12, I22

//     // stencils
//     tEigenArr<tVector2i>
//         mEdgeDiagVId; // the diag and anti-dian vertex of a edge
//     tEigenArr<tVector6i> mTriStencils; // check visualization in .cpp L2

//     tEigenArr<tVector9> mCijGrad[6];    // c00, c01, c11
//     tEigenArr<tMatrix9> mCijHessian[6]; // c00, c01, c11
//     // debug purpose
//     // tEigenArr<tVector6> mTriII_c;   // c used in II
//     // tEigenArr<tVector9> mTriII_t;   // c used in II
//     // tEigenArr<tMatrix69> mTriII_dcdx;   // dcdx in II
//     // tEigenArr<tMatrix9> mTriII_dtdx;   // dtdx in II

//     // for verify the derivative of T
//     tVector9 debug_Ti0u[3], debug_Ti1u[3];
//     tMatrix9 debug_dTi0udx[3];

//     // for debug
//     tVector9 debug_Tij0[2][3];
//     tMatrix9 debug_dTij0dx[2][3];

//     // tVector9 debug_Tij1[2][3];
//     // tMatrix9 debug_dtij1dx[2][3];

//     // for debug
//     tVector3 debug_gammai_u[3];
//     tMatrix39 debug_dgammaiu_dx[3];

//     // for verify the derivative of (Ti0 - Ti1) * Fj
//     tVector9 debug_MiTFj[6];
//     tMatrix9 debug_dMiTFjdx[9];

//     // functions
//     void InitStencilInfos();
//     void InitDminv(tEigenArr<tMatrix2> &array) const;
//     void CalcForms(const tVectorX &xcur, bool update_grad, bool update_hess);
//     void UpdateFirstForm(const tVectorX &xcur, int tid, const tMatrix32 &F,
//                          const tMatrix39 &dF0dx, const tMatrix39 &dF1dx,
//                          bool update_grad = true, bool update_hess = true);
//     void UpdateSecondForm(const tVectorX &xcur, int tid, const tMatrix32 &F,
//                           const tMatrix39 &dF0dx, const tMatrix39 &dF1dx,
//                           bool update_grad = true, bool update_hess = true);
// };
// SIM_DECLARE_CLASS_AND_PTR(cFForms);