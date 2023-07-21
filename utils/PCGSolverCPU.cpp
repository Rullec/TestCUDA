#include "utils/PCGSolverCPU.h"
#include "utils/DefUtil.h"

tVectorX cPCGSolverCPU::ApplyMatmul(const tSparseMat &A, const tVectorX &b,
                                    const std::vector<int> &fix_vertex_array)
{
    auto should_remove =
        [](const std::vector<int> &fix_vertex_array, int cur_idx)
    {
        for (auto &drag_pt_idx : fix_vertex_array)
        {
            bool is_return = (cur_idx == 4 * drag_pt_idx + 0) ||
                             (cur_idx == 4 * drag_pt_idx + 1) ||
                             (cur_idx == 4 * drag_pt_idx + 2) ||
                             (cur_idx == 4 * drag_pt_idx + 3);
            if (is_return)
                return true;
        }
        return false;
    };
    tVectorX res = tVectorX::Zero(b.size());
    OMP_PARALLEL_FOR(OMP_MAX_THREADS)
    for (int k = 0; k < A.outerSize(); ++k)
    {
        if (should_remove(fix_vertex_array, k))
            continue;
        for (tSparseMat::InnerIterator it(A, k); it; ++it)
        {
            // std::cout << it.row() << "\t";
            // std::cout << it.col() << "\t";
            // std::cout << it.value() << std::endl;
            if (should_remove(fix_vertex_array, it.col()))
            {
                // std::cout << "drag pt " << drag_pt << " ignore " << it.col()
                //           << std::endl;
                continue;
            }
            res[k] += b[it.col()] * it.value();
        }
    }
    return res;
}
void cPCGSolverCPU::Solve(const tSparseMat &A, const tVectorX &b, tVectorX &x,
                          float threshold, int &iters, float &residual,
                          int max_iters,
                          const std::vector<int> &fix_vertex_array /* = -1*/)
{
    // std::cout << "Eigen::nbThreads() = " << Eigen::nbThreads() << std::endl;
    int recalc_gap = 20;
    // tVectorX r = b - A * x;
    tVectorX r = b - cPCGSolverCPU::ApplyMatmul(A, x, fix_vertex_array);
    if (b.hasNaN())
    {
        printf("b has nan!\n");
        // exit(1);
    }
    for (auto fix_vertex : fix_vertex_array)
    {
        r.segment(3 * fix_vertex, 3).setZero();
        // std::cout << "r[init] = " << r.segment(3 * fix_vertex, 3).transpose()
        //           << std::endl;
        x.segment(3 * fix_vertex, 3).setZero();
    }
    tVectorX PInv = A.diagonal().cwiseInverse();
    tVectorX d = PInv.cwiseProduct(r);
    iters = 1;
    tVectorX Adi;
    tVectorX rnext;
    tVectorX PInvr, PInvrnext;
    while (iters < max_iters)
    {
        Adi.noalias() = cPCGSolverCPU::ApplyMatmul(A, d, fix_vertex_array);

        PInvr.noalias() = PInv.cwiseProduct(r);
        double deno = d.dot(Adi);
        if (deno == 0)
        {
            printf("[warn] deno = 0, break\n");
            break;
        }
        double alpha = r.dot(PInvr) / deno;

        x += alpha * d;
        rnext.noalias() = r - alpha * Adi;
        PInvrnext.noalias() = PInv.cwiseProduct(rnext);
        double beta = rnext.dot(PInvrnext) / (r.dot(PInvr));
        if (beta == 0)
        {
            break;
        }
        d = PInvrnext + beta * d;
        r.noalias() = rnext;
        iters += 1;
    }
    // printf("[cg] done, iters %d residual = %.2e\n", iters, r.norm());
    if (x.hasNaN())
    {
        printf("x has Nan\n");
        // exit(1);
    }
}

void cPCGSolverCPU::SolveBlockJacob(
    const tSparseMat &BlockJacInv, const tSparseMat &A, const tVectorX &b,
    tVectorX &x, float threshold, int &iters, float &residual, int max_iters,
    const std::vector<int> &fix_vertex_array /* = -1*/)
{
    int recalc_gap = 20;
    // tVectorX r = b - A * x;
    tVectorX r = b - cPCGSolverCPU::ApplyMatmul(A, x, fix_vertex_array);
    for (auto fix_vertex : fix_vertex_array)
    {
        r.segment(3 * fix_vertex, 3).setZero();
        x.segment(3 * fix_vertex, 3).setZero();
    }

    tVectorX d = BlockJacInv * r;
    iters = 1;
    tVectorX Adi;
    tVectorX rnext;
    tVectorX PInvr, PInvrnext;
    while (iters < max_iters)
    {
        Adi.noalias() = cPCGSolverCPU::ApplyMatmul(A, d, fix_vertex_array);

        PInvr.noalias() = BlockJacInv * r;
        double deno = d.dot(Adi);
        if (deno == 0)
        {
            printf("[warn] deno = 0, break\n");
            break;
        }
        double alpha = r.dot(PInvr) / deno;

        x += alpha * d;
        rnext.noalias() = r - alpha * Adi;
        PInvrnext.noalias() = BlockJacInv * rnext;
        double beta = rnext.dot(PInvrnext) / (r.dot(PInvr));
        if (beta == 0)
        {
            break;
        }
        d = PInvrnext + beta * d;
        r.noalias() = rnext;
        iters += 1;
    }
    printf("[block_pcg] done, iters %d residual = %.2e\n", iters, r.norm());
    if (x.hasNaN())
    {
        printf("x has Nan\n");
        // exit(1);
    }
}
