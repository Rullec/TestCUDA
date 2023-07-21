#pragma once
#include "utils/EigenUtil.h"
#include "utils/DefUtil.h"
#include <string>
#include <vector>

struct tTriangle;
struct tEdge;
struct tVertex;
struct tTet;
SIM_DECLARE_PTR(tTriangle);
SIM_DECLARE_PTR(tEdge);
SIM_DECLARE_PTR(tVertex);
SIM_DECLARE_PTR(tTet);

using tTrianglePtrVector = std::vector<tTrianglePtr>;
using tEdgePtrVector = std::vector<tEdgePtr>;
using tVertexPtrVector = std::vector<tVertexPtr>;
using tTetPtrVector = std::vector<tTetPtr>;
class cTetUtil
{
public:
    static void LoadTet(const std::string &path,
                        tVertexPtrVector &vertex_vec,
                        tEdgePtrVector &edge_vec,
                        tTrianglePtrVector &tri_vec,
                        tTetPtrVector &tet_vec);
    static _FLOAT CalculateTetVolume(
        const tVector4 &pos0,
        const tVector4 &pos1,
        const tVector4 &pos2,
        const tVector4 &pos3);
};