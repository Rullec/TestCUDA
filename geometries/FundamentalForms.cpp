// // clang-format off
// // ===============================triangle stencil===============================
// /*
//         ::                              ^!7~:                           ^?J55!  
//        7&J                            ^YJJYG&Y                         ~?^::?@? 
//      ^J7@J                            7:    G@^                        .    7G: 
//     7?.:@J                                 .B5                           .!5&G! 
//   ^Y!  ^@Y                                ^Y7                               :P@~
//   5PYYYP@B57                            :7!   ..                             ?#:
//        ^@Y                            :5BYJJJYY.                       ?GJ~^!J: 
//        .?~                            ~!!7!!!!:                         :^:..   
//         :GBPPPPPPPPPPPPPPPPPPPPPPPPPPP5PB#G555PPPPPPPPPPPPPPPPPPPPPPPP5GBJ      
//          ^#B^.........................:B&Y@Y..........................7@5.      
//           .G#^                       ^#B: !&5                        ?@?        
//             Y@7                     !@P.   ^#B:                     5&!         
//              7@Y                   J@J      .P&~                  :B#^          
//               ~&P.               .P@!         J@?                ~&P.           
//                :B#^             :##^           7@5              ?@J             
//                  5&!           !&G.             ^#G.           Y@7              
//                   |@J         ?@Y                .G#^        .G#^               
//               .::: ~&P.      5@7                   5@7      ^#G. :              
//             .55^^7P~:#B:   :B&~                     ?@J    7@5^!BB.             
//            .B&:   J@! P&~ ~&G:                       ~&P. J@?  .##.             
//            |@P    ^@B  J&5&P:.........................^#BP&~    ##.             
//            7@5    :@#   J@@P5PPPPPPPPPPPPPPPPPPPPPPPPP5B@&~     ##.             
//            :@B    ~@5    ^#B:                         ~&P.      ##              
//             7&?  :BG.     .G&~                       ?@J       :&&:             
//              :7!~7~         Y@7                     Y@7       :!??~:            
//                              7@Y                  .G#^                          
//                               ~&G.               ^#G.                           
//                                :G#^             7@5                             
//                                  5@!           J@?                              
//                            :555P5^?@J        .P&~                               
//                           ^G7^:::  ~&P.     ^#B:                                
//                           YBGBGJ.   :BB:   !&5                                  
//                             .^?&&^   .P&~ ?@J                                   
//                                ^@7     J&B&!                                    
//                          :^.   ?B:      !G^                                     
//                          7BG?!77.                        
// */
// // clang-format on

// #include "FundamentalForms.h"
// #include "geometries/DihedralAngles.h"
// #include "geometries/Primitives.h"
// #include "utils/LogUtil.h"
// #include "utils/ProfUtil.h"
// #include <iostream>
// const static tMatrix3 I3 = tMatrix3::Identity();
// extern tVectorX GetXPos(const std::vector<tVertexPtr> &mVertexArray);

// tVector3 FormToVec(const tMatrix2 &form)
// {
//     return tVector3(form(0, 0), form(0, 1), form(1, 1));
// }

// typedef Eigen::Matrix<_FLOAT, 6, 1> tVector6;
// typedef Eigen::Matrix<_FLOAT, 9, 1> tVector9;
// typedef Eigen::Matrix<_FLOAT, 18, 1> tVector18;
// typedef Eigen::Matrix<_FLOAT, 9, 18> tMatrix918;
// typedef Eigen::Matrix<_FLOAT, 18, 9> tMatrix189;
// template <class type, int m0, int n0, int m1, int n1>
// Eigen::Matrix<type, m0 * m1, n0 * n1>
// Hadamard(const Eigen::Matrix<type, m0, n0> &a,
//          const Eigen::Matrix<type, m1, n1> &b)
// {
//     using mat_type = Eigen::Matrix<type, m0 * m1, n0 * n1>;
//     mat_type mat = mat_type::Zero();
//     for (int i = 0; i < m0; i++)
//         for (int j = 0; j < n0; j++)
//         {
//             mat.block(i * m1, j * n1, m1, n1).noalias() = a(i, j) * b;
//         }
//     return mat;
// }
// cFForms::cFForms()
// {
//     mFirstArray.clear();
//     mSecondArray.clear();
// }

// void cFForms::Init(const std::vector<tVertexPtr> &v_array,
//                    const std::vector<tEdgePtr> &e_array,
//                    const std::vector<tTrianglePtr> &t_array)
// {
//     // 1. calculate stencil for each triangle (check the visualization at Line
//     // 2)
//     mVertexArray = v_array;
//     mEdgeArray = e_array;
//     mTrianglearray = t_array;

//     mNumOfE = e_array.size();
//     mNumOfV = v_array.size();
//     mNumOfT = t_array.size();

//     // 2. init dihedral angles for shape operator
//     mDihedralAngles = std::make_shared<cDihedralAngles>();
//     mDihedralAngles->Init(v_array, e_array, t_array);
//     mTriangleAreaStatic = mDihedralAngles->GetRawTriangleAreaArray();

//     for (int i = 0; i < 6; i++)
//     {
//         mCijGrad[i].resize(mNumOfT);
//         mCijHessian[i].resize(mNumOfT);
//     }
//     mEdgeLengthStatic.resize(mNumOfE, 0);
//     for (int i = 0; i < mNumOfE; i++)
//     {
//         mEdgeLengthStatic[i] = (mVertexArray[mEdgeArray[i]->mId0]->mPos -
//                                 mVertexArray[mEdgeArray[i]->mId1]->mPos)
//                                    .norm();
//     }

//     mBoundaryTriangle.resize(mNumOfT, false);
//     for (int i = 0; i < mNumOfT; i++)
//     {
//         mBoundaryTriangle[i] =
//             mEdgeArray[mTrianglearray[i]->mEId0]->mIsBoundary ||
//             mEdgeArray[mTrianglearray[i]->mEId1]->mIsBoundary ||
//             mEdgeArray[mTrianglearray[i]->mEId2]->mIsBoundary;
//     }
//     const auto &thetas = mDihedralAngles->GetAngles();
//     const auto &theta_grad = mDihedralAngles->GetGradAngles();
//     const auto &theta_hess = mDihedralAngles->GetHessAngles();

//     InitStencilInfos();
//     InitDminv(mDminv);
// }

// void cFForms::InitStencilInfos()
// {

//     // =========== edge oppo v ============
//     mEdgeDiagVId.resize(mNumOfE);

//     for (int e = 0; e < mNumOfE; e++)
//     {
//         auto cur_e = mEdgeArray[e];
//         tVector2i oppo_v = tVector2i::Constant(-1);
//         oppo_v[0] = mTrianglearray[cur_e->mTriangleId0]->SelectAnotherVertex(
//             cur_e->mId0, cur_e->mId1);
//         if (cur_e->mTriangleId1 != -1)
//         {
//             oppo_v[1] =
//                 mTrianglearray[cur_e->mTriangleId1]->SelectAnotherVertex(
//                     cur_e->mId0, cur_e->mId1);
//         };
//         mEdgeDiagVId[e] = oppo_v;
//         // std::cout << "edge " << e << " stencil = " << oppo_v.transpose()
//         //           << std::endl;
//     }

//     // =========== tri stencil ============
//     mTriStencils.resize(mNumOfT);

//     for (int t = 0; t < mNumOfT; t++)
//     {
//         tVector6i stencil = tVector6i::Constant(-1);
//         auto cur_t = mTrianglearray[t];

//         stencil[0] = cur_t->mId0;
//         stencil[1] = cur_t->mId1;
//         stencil[2] = cur_t->mId2;

//         for (int v = 0; v < 3; v++)
//         {
//             int oppo_v = stencil[v];
//             int oppo_eid = (v == 0) ? cur_t->mEId1
//                                     : ((v == 1) ? cur_t->mEId2 : cur_t->mEId0);
//             const auto cur_edge_oppo_info = mEdgeDiagVId[oppo_eid];
//             if (cur_edge_oppo_info[0] == oppo_v)
//             {
//                 stencil[3 + v] = cur_edge_oppo_info[1];
//             }
//             else if (cur_edge_oppo_info[1] == oppo_v)
//             {
//                 stencil[3 + v] = cur_edge_oppo_info[0];
//             }
//             else
//             {
//                 SIM_ERROR("cannot reach here.");
//             }
//         }
//         mTriStencils[t] = stencil;
//         // std::cout << "tri " << t << " stencil = " << stencil.transpose()
//         //           << std::endl;
//     }
// }

// void cFForms::Update(const tVectorX &xcur, bool update_grad, bool update_hess)
// {
//     cProfUtil::Begin("cloth/update_form");
//     cProfUtil::Begin("cloth/update_form/dihedrals");
//     mDihedralAngles->Update(xcur);
//     cProfUtil::End("cloth/update_form/dihedrals");
    
//     // mDihedralAngles->CheckGrad();

//     cProfUtil::Begin("cloth/update_form/calc_forms");
//     CalcForms(xcur, update_grad, update_hess);
//     cProfUtil::End("cloth/update_form/calc_forms");

//     cProfUtil::End("cloth/update_form");
//     // CheckFirstFormGrad();
//     // CheckFirstFormHess();
// }
// const std::vector<tVector3> &cFForms::GetI() const { return this->mFirstArray; }
// const std::vector<tVector3> &cFForms::GetII() const
// {
//     return this->mSecondArray;
// }

// void cFForms::InitDminv(tEigenArr<tMatrix2> &DminvArray) const
// {
//     DminvArray.resize(mNumOfT);
//     for (int i = 0; i < mNumOfT; i++)
//     {
//         auto tri = mTrianglearray[i];
//         auto v0 = mVertexArray[tri->mId0], v1 = mVertexArray[tri->mId1],
//              v2 = mVertexArray[tri->mId2];
//         DminvArray[i].col(0) = (v1->muv_simple - v0->muv_simple).cast<double>();
//         DminvArray[i].col(1) = (v2->muv_simple - v0->muv_simple).cast<double>();
//         DminvArray[i] = DminvArray[i].inverse().eval();
//     }
// }

// void cFForms::CalcForms(const tVectorX &xcur, bool update_grad,
//                         bool update_hess)
// {
//     cProfUtil::Begin("cloth/update_form/calc_forms/prepare");
//     if (update_hess)
//         SIM_ASSERT(update_grad);

//     mFirstArray.resize(mNumOfT);
//     mSecondArray.resize(mNumOfT);

//     mFirstGradArray.resize(mNumOfT);
//     mSecondGradArray.resize(mNumOfT);
//     // mTriII_c.resize(mNumOfT, tVector6::Zero());
//     // mTriII_dcdx.resize(mNumOfT, tMatrix69::Zero());
//     // mTriII_t.resize(mNumOfT, tVector9::Zero());
//     // mTriII_dtdx.resize(mNumOfT, tMatrix9::Zero());
//     for (int i = 0; i < 3; i++)
//     {
//         mFirstHessArray[i].resize(mNumOfT);
//         mSecondHessArray[i].resize(mNumOfT);
//     }

//     tEigenArr<tVector12> dthetadi = mDihedralAngles->GetGradAngles();
//     tMatrix32 S;
//     S << -1, -1, 1, 0, 0, 1;
//     cProfUtil::End("cloth/update_form/calc_forms/prepare");
//     cProfUtil::Begin("cloth/update_form/calc_forms/loop");
//     // OMP_PARALLEL_FOR(OMP_MAX_THREADS)
//     for (int i = 0; i < mNumOfT; i++)
//     {
//         cProfUtil::Begin("cloth/update_form/calc_forms/loop/prepare");
//         // std::cout << OMP_GET_NUM_THREAD_ID << " " << OMP_MAX_THREADS << std::endl;
//         auto cur_t = mTrianglearray[i];
//         tMatrix32 F;
//         // 1. calculate deformation gradient
//         const tVector3 &v0 = xcur.segment(3 * cur_t->mId0, 3),
//                        &v1 = xcur.segment(3 * cur_t->mId1, 3),
//                        &v2 = xcur.segment(3 * cur_t->mId2, 3);
//         F.col(0) = v1 - v0;
//         F.col(1) = v2 - v0;
//         F = (F * mDminv[i]).eval();
//         const tVector2 &Dminv0 = mDminv[i].col(0);
//         const tVector2 &Dminv1 = mDminv[i].col(1);

//         tMatrix39 dF0dx = Hadamard(tVector3(S * Dminv0), I3).transpose(),
//                   dF1dx = Hadamard(tVector3(S * Dminv1), I3).transpose();

//         cProfUtil::End("cloth/update_form/calc_forms/loop/prepare");
//         // 3. I form, II form
//         // std::cout << "tri " << i << " shape operator = \n"
//         //           << Lambda << std::endl;
//         cProfUtil::Begin("cloth/update_form/calc_forms/loop/I");

//         UpdateFirstForm(xcur, i, F, dF0dx, dF1dx, update_grad, update_hess);
//         cProfUtil::End("cloth/update_form/calc_forms/loop/I");
//         cProfUtil::Begin("cloth/update_form/calc_forms/loop/II");
//         UpdateSecondForm(xcur, i, F, dF0dx, dF1dx, update_grad, update_hess);
//         cProfUtil::End("cloth/update_form/calc_forms/loop/II");

//         // std::cout << "I = " << mFirstArray[i] << std::endl;
//         // std::cout << "II = " << mSecondArray[i] << std::endl;
//     }
//     cProfUtil::End("cloth/update_form/calc_forms/loop");
// }

// void cFForms::CheckFirstFormGrad()
// {
//     tVectorX xcur = GetXPos(mVertexArray);
//     _FLOAT eps = 1e-8;
//     for (int tid = 0; tid < mNumOfT; tid++)
//     {
//         int vid[3] = {mTrianglearray[tid]->mId0, mTrianglearray[tid]->mId1,
//                       mTrianglearray[tid]->mId2};
//         // 1. get analytic
//         tMatrix39 ana_grad = mFirstGradArray[tid];
//         tMatrix39 num_grad = tMatrix39::Zero();

//         // 2. get old form
//         tVector3 old_form = mFirstArray[tid];

//         // 3.
//         for (int j = 0; j < 9; j++)
//         {
//             xcur[3 * vid[j / 3] + j % 3] += eps;

//             {
//                 mDihedralAngles->Update(xcur);
//                 CalcForms(xcur, false, false);
//             }
//             tVector3 new_form = mFirstArray[tid];
//             num_grad.col(j) = (new_form - old_form) / eps;
//             xcur[3 * vid[j / 3] + j % 3] -= eps;
//         }

//         tMatrix39 diff_grad = (ana_grad - num_grad).cwiseAbs();
//         _FLOAT diff_max = diff_grad.maxCoeff();
//         printf("------- check tri %d form grad --------\n", tid);
//         std::cout << "ana_grad = \n" << ana_grad << std::endl;
//         std::cout << "num_grad = \n" << num_grad << std::endl;
//         std::cout << "diff_grad = \n" << diff_grad << std::endl;
//         if (diff_max > 1e-4)
//         {
//             std::cout << diff_max << std::endl;
//             exit(1);
//         }

//         // finish update
//         {
//             mDihedralAngles->Update(xcur);
//             CalcForms(xcur, false, false);
//         }
//     }
// }
// const tEigenArr<tMatrix39> &cFForms::GetIGradPerTriangle() const
// {
//     return this->mFirstGradArray;
// }
// const tEigenArr<tMatrix318> &cFForms::GetIIGradPerTriangle() const
// {
//     return this->mSecondGradArray;
// }

// const tEigenArr<tMatrix9> &cFForms::GetIHess(int idx) const
// {
//     SIM_ASSERT(idx >= 0 && idx <= 2);
//     return mFirstHessArray[idx];
// }

// const tEigenArr<tMatrix18> &cFForms::GetIIHess(int idx) const
// {
//     SIM_ASSERT(idx >= 0 && idx <= 2);
//     return mSecondHessArray[idx];
// }

// tEigenArr<tMatrix9> cFForms::GetIHessPerTri(int tid) const
// {
//     return {mFirstHessArray[0][tid], mFirstHessArray[1][tid],
//             mFirstHessArray[2][tid]};
// }
// tEigenArr<tMatrix18> cFForms::GetIIHessPerTri(int tid) const
// {
//     return {mSecondHessArray[0][tid], mSecondHessArray[1][tid],
//             mSecondHessArray[2][tid]};
// }
// void cFForms::CheckFirstFormHess()
// {
//     tVectorX xcur = GetXPos(mVertexArray);

//     _FLOAT eps = 1e-8;
//     for (int tid = 0; tid < mNumOfT; tid++)
//     {
//         int vid[3] = {mTrianglearray[tid]->mId0, mTrianglearray[tid]->mId1,
//                       mTrianglearray[tid]->mId2};
//         // check I11, I12, I22
//         for (int form_id = 0; form_id < 3; form_id++)
//         {
//             // 1. get analytic
//             tMatrix9 ana_hess = mFirstHessArray[form_id][tid];
//             tMatrix9 num_hess = tMatrix9::Zero();

//             // 2. get old form grad
//             tVector9 old_formgrad = mFirstGradArray[tid].row(form_id);

//             // 3.
//             for (int j = 0; j < 9; j++)
//             {
//                 xcur[3 * vid[j / 3] + j % 3] += eps;

//                 {
//                     mDihedralAngles->Update(xcur);
//                     CalcForms(xcur, true, false);
//                 }
//                 tVector9 new_formgrad = mFirstGradArray[tid].row(form_id);
//                 num_hess.col(j) = (new_formgrad - old_formgrad) / eps;
//                 xcur[3 * vid[j / 3] + j % 3] -= eps;
//             }

//             tMatrix9 diff_hess = (ana_hess - num_hess).cwiseAbs();
//             _FLOAT diff_max = diff_hess.maxCoeff();

//             if (diff_max > 1e-4)
//             {
//                 printf("------- check tri %d form %d hess --------\n", tid,
//                        form_id);
//                 std::cout << "ana_hess = \n" << ana_hess << std::endl;
//                 std::cout << "num_hess = \n" << num_hess << std::endl;
//                 std::cout << "diff_hess = \n" << diff_hess << std::endl;
//                 std::cout << diff_max << std::endl;
//                 exit(1);
//             }

//             // finish update
//             {
//                 mDihedralAngles->Update(xcur);
//                 CalcForms(xcur, true, false);
//             }
//         }
//         printf("tid %d check hess succ\n", tid);
//     }
// }
// void cFForms::UpdateFirstForm(const tVectorX &xcur, int tid, const tMatrix32 &F,
//                               const tMatrix39 &dF0dx, const tMatrix39 &dF1dx,
//                               bool update_grad, bool update_hess)
// {
//     tMatrix32 S;
//     S << -1, -1, 1, 0, 0, 1;
//     const tVector3 &F0 = F.col(0), F1 = F.col(1);
//     mFirstArray[tid] = FormToVec(F.transpose() * F);
//     if (update_grad)
//     {

//         // ================ grad I =============
//         /*
//         dI11/dx  = 2 * S * Dminv[:, 0] \otimes F0
//         dI22/dx  = 2 * S * Dminv[:, 1] \otimes F1
//         dI12/dx = S * Dminv[:, 0] \otimes F1 + S * Dminv[:, 1] \otimes F1
//         */

//         const tVector2 &Dminv0 = mDminv[tid].col(0);
//         const tVector2 &Dminv1 = mDminv[tid].col(1);
//         tVector3 tmp0 = 2 * S * Dminv0, tmp1 = 2 * S * Dminv1;

//         tVector9 res0 = Hadamard(tmp0, F0).transpose();
//         tVector9 res1 =
//             (Hadamard(tmp0, F1) + Hadamard(tmp1, F0)).transpose() / 2;
//         tVector9 res2 = Hadamard(tmp1, F1).transpose();
//         mFirstGradArray[tid].row(0) = res0;
//         mFirstGradArray[tid].row(1) = res1;
//         mFirstGradArray[tid].row(2) = res2;

//         // ================ hess I =============
//         if (update_hess)
//         {
//             /*
//             d^2 I11 / dx^2 = 2S Dminv[:, 0] \otimes dF0/dx

//             d^2 I22 / dx^2 = 2S Dminv[:, 1] \otimes dF1/dx

//             d^2 I12 / dx^2 =
//                 S Dminv[:, 0] \otimes dF1/dx
//                 +
//                 S Dminv[:, 1] \otimes dF0/dx
//             */

//             mFirstHessArray[0][tid] = Hadamard(tmp0, dF0dx);
//             mFirstHessArray[1][tid] =
//                 (Hadamard(tmp0, dF1dx) + Hadamard(tmp1, dF0dx)) / 2;
//             mFirstHessArray[2][tid] = Hadamard(tmp1, dF1dx);
//         }
//     }
// }
// #include "utils/MathUtil.h"
// extern tMatrix9 CalcDTDx(const tVector3 *v_array, tMatrix3 *dli_normed_dli,
//                          tMatrix39 &D, tMatrix39 &K);
// extern void CalcDTDx_times_u_debug(const tVector3 *v_array, const tVector3 &u,
//                                    tVector9 *debug_Ti0, tVector9 *debug_Ti1);
// tMatrix39 GetCi(int idx);
// tVector3 u_debug = tVector3::Ones();

// // int tid_debug = -1;

// tMatrix3 D_dxnormed_dx_times_u_dx(_FLOAT x_norm, const tMatrix3 &gamma,
//                                   const tVector3 &u, const tVector3 &x_normed)
// {
//     _FLOAT x_norm_inv = 1.0 / x_norm;
//     return -x_norm_inv * (gamma * u * x_normed.transpose() +
//                           (x_normed.transpose() * u * tMatrix3::Identity() +
//                            x_normed * u.transpose()) *
//                               gamma);
// }
// void cFForms::UpdateSecondForm(const tVectorX &xcur, int tid,
//                                const tMatrix32 &F, const tMatrix39 &dF0dx,
//                                const tMatrix39 &dF1dx, bool update_grad,
//                                bool update_hess)
// {
//     // if (mBoundaryTriangle[tid])
//     //     return;
//     auto edge_angles = mDihedralAngles->GetAngles();
//     /*
//            2. shape operator

//            \Lambda =
//                \sum_i
//                    \theta_i * |li'|/ (2 * A')
//                    *
//                    (ti  * ti.T)
//            For simplicity, we use static length and area in the formula
//            where ti =
//                    (n \times li)
//                    /
//                    |n \times li|
//        */
//     auto cur_t = mTrianglearray[tid];
//     // 1. calculate deformation gradient
//     const tVector3 v[3] = {
//         xcur.segment(3 * cur_t->mId0, 3),
//         xcur.segment(3 * cur_t->mId1, 3),
//         xcur.segment(3 * cur_t->mId2, 3),
//     };

//     int eid[3] = {cur_t->mEId0, cur_t->mEId1, cur_t->mEId2};

//     tVector3 edge_vec[3] = {v[1] - v[0], v[2] - v[1], v[0] - v[2]};
//     tVector3 n_raw = edge_vec[0].cross(edge_vec[1]);
//     _FLOAT n_norm = n_raw.norm();
//     tVector3 n_normed = n_raw / n_norm;
//     tMatrix3 gamma_n =
//         (tMatrix3::Identity() - n_normed * n_normed.transpose()) / n_norm;
//     tMatrix3 skew_n_normed = cMathUtil::VectorToSkewMat(n_normed);
//     _FLOAT edge_norm[3] = {edge_vec[0].norm(), edge_vec[1].norm(),
//                            edge_vec[2].norm()};
//     tVector3 edge_vec_normed[3] = {edge_vec[0] / edge_norm[0],
//                                    edge_vec[1] / edge_norm[1],
//                                    edge_vec[2] / edge_norm[2]};
//     tMatrix3 edge_vec_normed_skew[3] = {
//         cMathUtil::VectorToSkewMat(edge_vec_normed[0]),
//         cMathUtil::VectorToSkewMat(edge_vec_normed[1]),
//         cMathUtil::VectorToSkewMat(edge_vec_normed[2])};
//     const tVector3 &F0 = F.col(0), F1 = F.col(1);
//     tMatrix3 Lambda = tMatrix3::Zero();
//     tMatrix32 c = tMatrix32::Zero(); // coef, cij = t_i'.dot(Fj)

//     // mTriII_dtdx[tid] = M_dtidx;
//     tVector6 cij = tVector6::Zero();

//     tVector3 t[3] = {};
//     bool edge_is_boudnary[3] = {false, false, false};
//     for (int i = 0; i < 3; i++)
//     {
//         t[i] = edge_vec[i].cross(n_normed);
//         t[i].normalize();

//         // mTriII_t[tid].segment(3 * i, 3) = ti;

//         // mTriII_c[tid][2 * i + 0] = ti.dot(F0);
//         // mTriII_c[tid][2 * i + 1] = ti.dot(F1);
//         cij[2 * i + 0] = t[i].dot(F0);
//         cij[2 * i + 1] = t[i].dot(F1);

//         // ignore boundary edge

//         edge_is_boudnary[i] = mEdgeArray[eid[i]]->mIsBoundary;
//         if (edge_is_boudnary[i] == false)
//         {
//             _FLOAT theta = edge_angles[eid[i]];

//             Lambda += theta * mEdgeLengthStatic[eid[i]] /
//                       (2 * mTriangleAreaStatic[tid]) * t[i] * t[i].transpose();
//             // Lambda += mEdgeLengthStatic[eid[i]] /
//             //           (2 * mTriangleAreaStatic[tid]) * t[i] *
//             //           t[i].transpose();
//         }
//     }

//     mSecondArray[tid] = FormToVec(F.transpose() * Lambda * F);

//     if (update_grad)
//     {
//         tMatrix3 gammai_dlinormed_dli[3];
//         tMatrix39 dndx;
//         tMatrix39 K_edge_skew_l1_l2_l0;
//         tMatrix9 M_dtidx =
//             CalcDTDx(v, gammai_dlinormed_dli, dndx, K_edge_skew_l1_l2_l0);

//         // if (tid == tid_debug)
//         // {
//         //     // calculate Tij0
//         //     for (int i = 0; i < 3; i++)
//         //     {
//         //         debug_gammai_u[i] = gammai_dlinormed_dli[i] * u_debug;
//         //     }
//         //     CalcDTDx_times_u_debug(v, u_debug, debug_Ti0u, debug_Ti1u);

//         //     for (int j = 0; j < 2; j++)
//         //     {
//         //         for (int i = 0; i < 3; i++)
//         //         {
//         //             // calc Tij0
//         //             const tMatrix39 &Ci = GetCi(i);
//         //             debug_Tij0[j][i] = Ci.transpose() *
//         //                                gammai_dlinormed_dli[i] *
//         //                                skew_n_normed * (j == 0 ? F0 : F1);
//         //         }
//         //     }
//         // }

//         mSecondGradArray[tid].setZero();
//         for (int i = 0; i < 3; i++)
//         {

//             tMatrix39 Mi = M_dtidx.block(3 * i, 0, 3, 9);

//             mCijGrad[2 * i + 0][tid] =
//                 Mi.transpose() * F0 + dF0dx.transpose() * t[i];
//             mCijGrad[2 * i + 1][tid] =
//                 Mi.transpose() * F1 + dF1dx.transpose() * t[i];
//             // if (mCijGrad[2 * i + 0][tid].hasNaN())
//             // {
//             //     std::cout << "mCijGrad[2 * i + 0][tid].hasNaN() = "
//             //               << mCijGrad[2 * i + 0][tid].transpose() <<
//             //               std::endl;
//             //     exit(1);
//             // }
//             // if (mCijGrad[2 * i + 1][tid].hasNaN())
//             // {
//             //     std::cout << "mCijGrad[2 * i + 1][tid].hasNaN() = "
//             //               << mCijGrad[2 * i + 1][tid].transpose() <<
//             //               std::endl;
//             //     exit(1);
//             // }
//             debug_MiTFj[2 * i + 0] = Mi.transpose() * F0;
//             debug_MiTFj[2 * i + 1] = Mi.transpose() * F1;
//         }
//         // 1. get and rearrange dtheta/dx
//         auto ExpandByStencil = [](const tVector4i &theta_stencil,
//                                   const tVector12 &DthetaDx,
//                                   const tVector6i &tri_stencil)
//         {
//             tVector18 res = tVector18::Zero();
//             for (int theta_vid = 0; theta_vid < 4; theta_vid++)
//             {
//                 int tri_vid = 0;
//                 while (tri_stencil[tri_vid] != theta_stencil[theta_vid])
//                     tri_vid += 1;

//                 res.segment(3 * tri_vid, 3) =
//                     DthetaDx.segment(3 * theta_vid, 3);
//             }
//             return res;
//         };

//         tVector18 DThetaDx[3];

//         for (int e = 0; e < 3; e++)
//         {
//             tVector12 DthetaDx = mDihedralAngles->GetGradAngles()[eid[e]];
//             tVector4i DthetaDx_stencil = mDihedralAngles->GetStencil(eid[e]);
//             tVector6i Dtri_stencil = mTriStencils[tid];
//             DThetaDx[e] =
//                 ExpandByStencil(DthetaDx_stencil, DthetaDx, Dtri_stencil);
//             // std::cout << "dtheta_dx for e " << e << " = "
//             //           << DThetaDx[e].transpose() << std::endl;
//         }

//         tVector2i indices_array[3] = {{0, 0}, {0, 1}, {1, 1}};
//         for (int ab_id = 0; ab_id < 3; ab_id++)
//         {
//             int a = indices_array[ab_id][0], b = indices_array[ab_id][1];
//             for (int i = 0; i < 3; i++)
//             {
//                 _FLOAT pref =
//                     mEdgeLengthStatic[eid[i]] / (2 * mTriangleAreaStatic[tid]);
//                 // if (DThetaDx[i].hasNaN())
//                 // {
//                 //     std::cout << " DThetaDx[i] =  " <<
//                 //     DThetaDx[i].transpose()
//                 //               << std::endl;
//                 //     exit(1);
//                 // }
//                 mSecondGradArray[tid].row(ab_id) +=
//                     pref * cij[2 * i + a] * cij[2 * i + b] * DThetaDx[i];
//                 mSecondGradArray[tid].row(ab_id).segment(0, 9) +=
//                     pref * edge_angles[eid[i]] *
//                     (cij[2 * i + a] * mCijGrad[2 * i + b][tid] +
//                      cij[2 * i + b] * mCijGrad[2 * i + a][tid]);
//                 // mSecondGradArray[tid].row(ab_id).segment(0, 9) +=
//                 //     pref * (cij[2 * i + a] * dcijdx[2 * i + b] +
//                 //             cij[2 * i + b] * dcijdx[2 * i + a]);
//             }
//         }
//         // if (mSecondGradArray[tid].hasNaN())
//         // {
//         //     std::cout << "mSecondGradArray[tid] has Nan = \n"
//         //               << mSecondGradArray[tid] << std::endl;
//         //     exit(1);
//         // }
//         if (update_hess)
//         {
//             // clang-format off

//             // please check the note "第一二基本形式离散形式计算grad/hess", formula (8.3)
//             // clang-format on
//             // if (tid == tid_debug)
//             // {
//             // for (int i = 0; i < 3; i++)
//             // {
//             //     _FLOAT li_norm = edge_norm[i];
//             //     const tVector3 &li_normed = edge_vec_normed[i];
//             //     _FLOAT inv_li_norm = 1.0 / li_norm;
//             //     const tMatrix39 Ci = GetCi(i);
//             //     // 1. calculate p1
//             //     const tMatrix3 &gammai = gammai_dlinormed_dli[i];
//             //     tMatrix3 p1 = gammai * skew_n_normed * u_debug *
//             //                   li_normed.transpose();
//             //     tMatrix3 p2 =
//             //         (li_normed.transpose() * skew_n_normed * u_debug *
//             //              tMatrix3::Identity() -
//             //          li_normed * u_debug.transpose() * skew_n_normed) *
//             //         gammai;
//             //     tMatrix39 part81 = -inv_li_norm * (p1 + p2) * Ci;
//             //     tMatrix39 part82 =
//             //         -gammai * cMathUtil::VectorToSkewMat(u_debug) * dndx;

//             //     debug_dTi0udx[i] = Ci.transpose() * (part81 + part82);
//             // }

//             // for (int j = 0; j < 2; j++)
//             // {
//             //     const tVector3 &Fj = (j == 0) ? F0 : F1;

//             //     const tMatrix39 &dFjdx = (j == 0) ? dF0dx : dF1dx;
//             //     for (int i = 0; i < 3; i++)
//             //     {
//             //         _FLOAT li_norm = edge_norm[i];
//             //         const tVector3 &li_normed = edge_vec_normed[i];
//             //         _FLOAT inv_li_norm = 1.0 / li_norm;
//             //         const tMatrix39 Ci = GetCi(i);
//             //         // 1. calculate p1
//             //         const tMatrix3 &gammai = gammai_dlinormed_dli[i];

//             //         tMatrix39 part81 = D_dxnormed_dx_times_u_dx(
//             //                                li_norm, gammai,
//             //                                skew_n_normed * Fj, li_normed) *
//             //                            Ci;
//             //         tMatrix39 part82 =
//             //             -gammai * cMathUtil::VectorToSkewMat(Fj) * dndx;

//             //         debug_dTij0dx[j][i] =
//             //             Ci.transpose() *
//             //             (part81 + part82 + gammai * skew_n_normed * dFjdx);
//             //     }
//             // }

//             for (int i = 0; i < 3; i++)
//             {
//                 _FLOAT li_norm = edge_norm[i];
//                 const tVector3 &li_normed = edge_vec_normed[i];
//                 _FLOAT inv_li_norm = 1.0 / li_norm;
//                 const tMatrix39 Ci = GetCi(i);
//                 // 1. calculate p1
//                 const tMatrix3 &gammai = gammai_dlinormed_dli[i];

//                 for (int j = 0; j < 2; j++)
//                 {
//                     const tVector3 &Fj = (j == 0) ? F0 : F1;

//                     const tMatrix39 &dFjdx = (j == 0) ? dF0dx : dF1dx;

//                     tMatrix39 part81 =
//                         D_dxnormed_dx_times_u_dx(
//                             li_norm, gammai, skew_n_normed * Fj, li_normed) *
//                         Ci;
//                     tMatrix39 part82 =
//                         -gammai * cMathUtil::VectorToSkewMat(Fj) * dndx;

//                     tMatrix9 MiTdFjdx =
//                         M_dtidx.block(3 * i, 0, 3, 9).transpose() * dFjdx;
//                     mCijHessian[2 * i + j][tid] =
//                         Ci.transpose() *
//                             (part81 + part82 + gammai * skew_n_normed * dFjdx) +
//                         MiTdFjdx.transpose();
//                 }
//             }
//             /*

//             Hess II_{ab} =
//                 \sum_{i} l_i' / (2 A') * [p1 + p1.T + p2 + p3]

//             p1 =  ( c_ia  dc_{ib}/dx + c_{ib} * dc_{ia}/dx) *
//             (d\theta_i/dx)^T

//             p2 = (dc_{ia}/dx * (dc_{ib}/dx)^T + dc_{ib}/dx *
//             (dc_{ia}/dx) ^T) * theta_i

//             p3 = (c_{ia} * Hess c_{ib} + c_{ib} *
//             Hess c_{ia} ) * theta_i
//             */

//             for (int i = 0; i < 3; i++)
//             {
//                 mSecondHessArray[i][tid].setZero();
//                 if (edge_is_boudnary[i] == true)
//                     continue;
//                 _FLOAT thetai = edge_angles[eid[i]];
//                 const tVector18 &dtheta_dxi = DThetaDx[i];
//                 _FLOAT pref =
//                     mEdgeLengthStatic[eid[i]] / (2 * mTriangleAreaStatic[tid]);

//                 for (int ab_id = 0; ab_id < 3; ab_id++)
//                 {
//                     int a = indices_array[ab_id][0],
//                         b = indices_array[ab_id][1];

//                     _FLOAT cia = cij[2 * i + a], cib = cij[2 * i + b];
//                     const tVector9 &dcia = mCijGrad[2 * i + a][tid],
//                                    &dcib = mCijGrad[2 * i + b][tid];

//                     const tMatrix9 &hess_ia = mCijHessian[2 * i + a][tid];
//                     const tMatrix9 &hess_ib = mCijHessian[2 * i + b][tid];

//                     tMatrix918 p1 =
//                         (cia * dcib + cib * dcia) * dtheta_dxi.transpose();
//                     tMatrix189 p1T = p1.transpose();

//                     tMatrix9 p2 =
//                         (dcia * dcib.transpose() + dcib * dcia.transpose()) *
//                         thetai;
//                     tMatrix9 p3 = (cia * hess_ib + cib * hess_ia) * thetai;

//                     mSecondHessArray[i][tid].block(0, 0, 9, 18) += p1;
//                     mSecondHessArray[i][tid].block(0, 0, 18, 9) += p1T;

//                     mSecondHessArray[i][tid].block(0, 0, 9, 9) += p2 + p3;
//                 }
//                 mSecondHessArray[i][tid] *= pref;
//                 if (mSecondHessArray[i][tid].hasNaN())
//                 {
//                     std::cout << "mSecondHessArray[i][tid] = \n"
//                               << mSecondHessArray[i][tid] << std::endl;
//                     exit(1);
//                 }
//             }

//             // // calculate dgammaiu/dx
//             // for (int i = 0; i < 3; i++)
//             // {
//             //     _FLOAT li_norm = edge_norm[i];
//             //     const tVector3 &li_normed = edge_vec_normed[i];
//             //     _FLOAT inv_li_norm = 1.0 / li_norm;
//             //     const tMatrix39 Ci = GetCi(i);
//             //     // 1. calculate p1
//             //     const tMatrix3 &gammai = gammai_dlinormed_dli[i];
//             //     tMatrix3 p1 = gammai * u_debug * li_normed.transpose();
//             //     tMatrix3 p2 = (li_normed.transpose() * u_debug *
//             //                        tMatrix3::Identity() +
//             //                    li_normed * u_debug.transpose()) *
//             //                   gammai;

//             //     debug_dgammaiu_dx[i] = -inv_li_norm * (p1 + p2) * Ci;
//             // }
//             // }
//         }
//     }
// }

// // void cFForms::CheckSecondFormCGrad()
// // {

// //     _FLOAT eps = 1e-8;
// //     for (int tid = 0; tid < mNumOfT; tid++)
// //     {

// //         int vid[3] = {mTrianglearray[tid]->mId0, mTrianglearray[tid]->mId1,
// //                       mTrianglearray[tid]->mId2};
// //         // 1. get analytic
// //         tMatrix69 ana_grad = mTriII_dcdx[tid];
// //         tMatrix69 num_grad = tMatrix69::Zero();

// //         // 2. get old form
// //         tVector6 old_t = mTriII_c[tid];

// //         // 3.
// //         for (int j = 0; j < 9; j++)
// //         {
// // 3 *              xcur[vid[j / 3] + j % 3] += eps;

// //             {
// //                 mDihedralAngles->Update();
// //                 CalcForms(false, false);
// //             }
// //             tVector6 new_t = mTriII_c[tid];
// //             num_grad.col(j) = (new_t - old_t) / eps;
// // 3 *              xcur[vid[j / 3] + j % 3] -= eps;
// //         }

// //         tMatrix69 diff_grad = (ana_grad - num_grad).cwiseAbs();
// //         _FLOAT diff_max = diff_grad.maxCoeff();

// //         if (diff_max > 1e-4)
// //         {
// //             printf("------- check tri %d II form c grad --------\n", tid);
// //             std::cout << "ana_grad = \n" << ana_grad << std::endl;
// //             std::cout << "num_grad = \n" << num_grad << std::endl;
// //             std::cout << "diff_grad = \n" << diff_grad << std::endl;
// //             std::cout << diff_max << std::endl;
// //             exit(1);
// //             // exit(1);
// //         }

// //         // finish update
// //         {
// //             mDihedralAngles->Update();
// //             CalcForms(false, false);
// //         }
// //         printf("check d II c /dx succ for tri %d\n", tid);
// //     }
// // }

// void cFForms::CheckSecondFormGrad()
// {
//     tVectorX xcur = GetXPos(mVertexArray);
//     _FLOAT eps = 1e-8;
//     for (int tid = 0; tid < mNumOfT; tid++)
//     {
//         if (mBoundaryTriangle[tid])
//             continue;
//         tVector6i vid = this->mTriStencils[tid];
//         // 1. get analytic
//         tMatrix318 ana_grad = mSecondGradArray[tid];
//         tMatrix318 num_grad = tMatrix318::Zero();

//         // 2. get old form
//         tVector3 old_form = mSecondArray[tid];
//         if (old_form.cwiseAbs().minCoeff() < 1e-2)
//         {
//             printf("too small form, ignore check\n");
//             continue;
//         }
//         // 3.
//         for (int j = 0; j < 3 * vid.size(); j++)
//         {
//             xcur[3 * vid[j / 3] + j % 3] += eps;

//             {
//                 mDihedralAngles->Update(xcur, false, false);
//                 CalcForms(xcur, false, false);
//             }
//             tVector3 new_form = mSecondArray[tid];
//             num_grad.col(j) = (new_form - old_form) / eps;
//             xcur[3 * vid[j / 3] + j % 3] -= eps;
//         }

//         tMatrix318 diff_grad = (ana_grad - num_grad).cwiseAbs();
//         _FLOAT diff_max = diff_grad.maxCoeff();

//         if (diff_max > 10)
//         {
//             printf("------- check tri %d second form grad --------\n", tid);
//             std::cout << "form = " << old_form.transpose() << std::endl;
//             std::cout << "ana_grad = \n" << ana_grad << std::endl;
//             std::cout << "num_grad = \n" << num_grad << std::endl;
//             std::cout << "diff_grad = \n" << diff_grad << std::endl;
//             std::cout << diff_max << std::endl;
//             exit(1);
//         }
//         printf("check tri %d second form grad succ, diff_max = %.2f, deriv max "
//                "%.2f\n",
//                tid, diff_max, ana_grad.cwiseAbs().maxCoeff());
//         // finish update
//         {
//             mDihedralAngles->Update(xcur);
//             CalcForms(xcur, false, false);
//         }
//     }
// }

// void cFForms::CheckSecondFormTi0uGrad()
// {
//     tVectorX xcur = GetXPos(mVertexArray);
//     _FLOAT eps = 1e-6;
//     // 0. we first verify Ti0 now
//     for (int tid = 0; tid < this->mNumOfT; tid++)
//     {
//         // 1. set tid
//         // tid_debug = tid;
//         Update(xcur, true, true);

//         int vid[3] = {mTrianglearray[tid]->mId0, mTrianglearray[tid]->mId1,
//                       mTrianglearray[tid]->mId2};
//         tVector9 old_Ti0[3] = {debug_Ti0u[0], debug_Ti0u[1], debug_Ti0u[2]};
//         for (int i = 0; i < 3; i++)
//         {

//             // 2. get old Ti0 (9 * 1)

//             // 3. get analytic dTi0/dx (9 * 9)
//             tMatrix9 ana_dTi0dx = debug_dTi0udx[i];
//             tMatrix9 num_dTi0udx = tMatrix9::Zero();
//             // 4. change x, get new Tij, calculate numerical dTij/dx
//             for (int dof = 0; dof < 9; dof++)
//             {
//                 xcur[3 * vid[dof / 3] + dof % 3] += eps;
//                 Update(xcur, true, false);
//                 tVector9 new_Ti0 = debug_Ti0u[i];
//                 num_dTi0udx.col(dof) = (new_Ti0 - old_Ti0[i]) / eps;

//                 xcur[3 * vid[dof / 3] + dof % 3] -= eps;
//             }
//             // 5.
//             tMatrix9 diff = (num_dTi0udx - ana_dTi0dx).cwiseAbs();
//             _FLOAT max_diff = diff.maxCoeff();
//             if (max_diff > 1e-1)
//             {
//                 printf("verify T%d*u for tri %d failed\n", i, tid);
//                 std::cout << "ana = \n" << ana_dTi0dx << std::endl;
//                 std::cout << "num = \n" << num_dTi0udx << std::endl;
//                 std::cout << "diff = \n" << diff << std::endl;
//                 std::cout << "max_diff = " << max_diff << std::endl;
//                 exit(1);
//             }
//             printf("verify Ti0(u) for tri %d succ dof %d\n", tid, i);
//         }
//     }
// }

// void cFForms::CheckSecondFormGammaiUGrad()
// {
//     tVectorX xcur = GetXPos(mVertexArray);
//     int part = 0;
//     _FLOAT eps = 1e-6;
//     // 0. we first verify Ti0 now
//     for (int tid = 0; tid < this->mNumOfT; tid++)
//     {
//         // 1. set tid
//         // tid_debug = tid;
//         Update(xcur, true, true);

//         int vid[3] = {mTrianglearray[tid]->mId0, mTrianglearray[tid]->mId1,
//                       mTrianglearray[tid]->mId2};
//         tVector3 old_Ti0[3] = {debug_gammai_u[0], debug_gammai_u[1],
//                                debug_gammai_u[2]};
//         for (int i = 0; i < 3; i++)
//         {

//             // 2. get old Ti0 (9 * 1)

//             // 3. get analytic dTi0/dx (9 * 9)
//             tMatrix39 ana_dTi0dx = debug_dgammaiu_dx[i];
//             tMatrix39 num_dTi0udx = tMatrix39::Zero();
//             // 4. change x, get new Tij, calculate numerical dTij/dx
//             for (int dof = 0; dof < 9; dof++)
//             {
//                 xcur[3 * vid[dof / 3] + dof % 3] += eps;
//                 Update(xcur, true, false);
//                 tVector3 new_Ti0 = debug_gammai_u[i];
//                 num_dTi0udx.col(dof) = (new_Ti0 - old_Ti0[i]) / eps;

//                 xcur[3 * vid[dof / 3] + dof % 3] -= eps;
//             }
//             // 5.
//             tMatrix39 diff = (num_dTi0udx - ana_dTi0dx).cwiseAbs();
//             _FLOAT max_diff = diff.maxCoeff();
//             if (max_diff > 1e-1)
//             {
//                 printf("verify dgamma%du/dx for tri %d failed\n", i, tid);
//                 std::cout << "ana = \n" << ana_dTi0dx << std::endl;
//                 std::cout << "num = \n" << num_dTi0udx << std::endl;
//                 std::cout << "diff = \n" << diff << std::endl;
//                 std::cout << "max_diff = " << max_diff << std::endl;
//                 exit(1);
//             }
//             printf("verify dgamma%du/dx for tri %d succ\n", i, tid);
//         }
//     }
// }

// void cFForms::CheckSecondFormTij0Grad()
// {
//     auto xcur = GetXPos(mVertexArray);
//     int part = 0;
//     _FLOAT eps = 1e-6;
//     // 0. we first verify Ti0 now
//     for (int tid = 0; tid < this->mNumOfT; tid++)
//     {
//         // 1. set tid
//         // tid_debug = tid;
//         Update(xcur, true, true);

//         int vid[3] = {mTrianglearray[tid]->mId0, mTrianglearray[tid]->mId1,
//                       mTrianglearray[tid]->mId2};
//         tVector9 old_Tij0[2][3];
//         for (int j = 0; j < 2; j++)
//         {
//             for (int i = 0; i < 3; i++)
//             {
//                 old_Tij0[j][i] = debug_Tij0[j][i];
//             }
//         }

//         for (int j = 0; j < 2; j++)
//         {
//             for (int i = 0; i < 3; i++)
//             {

//                 // 2. get old Ti0 (9 * 1)

//                 // 3. get analytic dTi0/dx (9 * 9)
//                 tMatrix9 ana_dTij0dx = debug_dTij0dx[j][i];
//                 tMatrix9 num_dTij0udx = tMatrix9::Zero();
//                 // 4. change x, get new Tij, calculate numerical dTij/dx
//                 for (int dof = 0; dof < 9; dof++)
//                 {
//                     xcur[3 * vid[dof / 3] + dof % 3] += eps;
//                     Update(xcur, true, false);

//                     num_dTij0udx.col(dof) =
//                         (debug_Tij0[j][i] - old_Tij0[j][i]) / eps;

//                     xcur[3 * vid[dof / 3] + dof % 3] -= eps;
//                 }
//                 // 5.
//                 tMatrix9 diff = (num_dTij0udx - ana_dTij0dx).cwiseAbs();
//                 _FLOAT max_diff = diff.maxCoeff();
//                 if (max_diff > 1e-1)
//                 {
//                     printf("verify dT%d%d0/dx for tri %d failed\n", i, j, tid);
//                     std::cout << "ana = \n" << ana_dTij0dx << std::endl;
//                     std::cout << "num = \n" << num_dTij0udx << std::endl;
//                     std::cout << "diff = \n" << diff << std::endl;
//                     std::cout << "max_diff = " << max_diff << std::endl;
//                     exit(1);
//                 }
//                 printf("verify dT%d%d0/dx for tri %d succ\n", i, j, tid);
//             }
//         }
//     }
// }

// void cFForms::CheckSecondFormCijHess()
// {
//     auto xcur = GetXPos(mVertexArray);
//     _FLOAT eps = 1e-6;
//     for (int tid = 0; tid < mNumOfT; tid++)
//     {
//         // check cij[id] for triangle tid

//         for (int id = 0; id < 6; id++)
//         {

//             Update(xcur, true, true);

//             int vid[3] = {mTrianglearray[tid]->mId0, mTrianglearray[tid]->mId1,
//                           mTrianglearray[tid]->mId2};

//             tVector9 old_cij_grad = this->mCijGrad[id][tid];
//             tMatrix9 ana_cij_hess = this->mCijHessian[id][tid];
//             tMatrix9 num_cij_hess = tMatrix9::Zero();
//             for (int i = 0; i < 9; i++)
//             {
//                 xcur[3 * vid[i / 3] + i % 3] += eps;
//                 Update(xcur, true, false);
//                 num_cij_hess.col(i) = (mCijGrad[id][tid] - old_cij_grad) / eps;

//                 xcur[3 * vid[i / 3] + i % 3] -= eps;
//             }
//             tMatrix9 diff = (num_cij_hess - ana_cij_hess).cwiseAbs();
//             _FLOAT max_diff = diff.maxCoeff();
//             if (max_diff > 1e-1)
//             {
//                 printf("verify cij(%d) hess for tri %d failed\n", id, tid);
//                 std::cout << "ana = \n" << ana_cij_hess << std::endl;
//                 std::cout << "num = \n" << num_cij_hess << std::endl;
//                 std::cout << "diff = \n" << diff << std::endl;
//                 std::cout << "max_diff = " << max_diff << std::endl;
//                 exit(1);
//             }
//             else
//             {
//                 printf("verify cij(%d) hess for tri %d succ\n", id, tid);
//             }
//         }
//     }
// }

// const tEigenArr<tVector6i> &cFForms::GetStencil() const
// {
//     return this->mTriStencils;
// }

// const std::vector<_FLOAT> &cFForms::GetTriangleAreaStatic() const
// {
//     return this->mTriangleAreaStatic;
// }