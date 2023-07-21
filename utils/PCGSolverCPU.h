#pragma once
#include "utils/EigenUtil.h"
#include "utils/SparseUtil.h"
class cPCGSolverCPU
{
public:
    static tVectorX ApplyMatmul(const tSparseMat &A, const tVectorX &b,
                                const std::vector<int> &fix_vertex_array);
    static void Solve(const tSparseMat &A, const tVectorX &b, tVectorX &x,
                      float threshold, int &iters, float &residual,
                      int max_iters,
                      const std::vector<int> &fix_vertex_array /* = -1*/);
    static void
    SolveBlockJacob(const tSparseMat & BlockJacInv, const tSparseMat &A, const tVectorX &b, tVectorX &x,
                    float threshold, int &iters, float &residual, int max_iters,
                    const std::vector<int> &fix_vertex_array /* = -1*/);
};