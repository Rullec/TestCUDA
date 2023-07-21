#pragma once
#include "geometries/Primitives.h"
using tMatrix12 = Eigen::Matrix<_FLOAT, 12, 12>;
using tVector12 = Eigen::Matrix<_FLOAT, 12, 1>;

class cDihedralAngles
{
public:
    explicit cDihedralAngles();
    virtual void Init(const std::vector<tVertexPtr> &v_array,
                      const std::vector<tEdgePtr> &e_array,
                      const std::vector<tTrianglePtr> &t_array);
    virtual void Update(const tVectorX &xcur, bool update_grad = true,
                        bool update_hess = true);
    const std::vector<_FLOAT> GetAngles() const;
    const tEigenArr<tVector12> GetGradAngles() const;
    const tEigenArr<tMatrix12> GetHessAngles() const;
    const std::vector<_FLOAT> &GetRawTriangleAreaArray() const;
    void CheckGrad();
    void CheckHess();
    const tVector4i GetStencil(int) const;

protected:
    std::vector<tVertexPtr> mVertexArray;
    std::vector<tEdgePtr> mEdgeArray;
    std::vector<tTrianglePtr> mTriangleArray;
    int mNumOfVertices, mNumOfEdges, mNumOfTriangles;

    // ========= static vars ============
    std::vector<_FLOAT> mRawEdgeLengthArray;
    tEigenArr<tVector2> mRawHeightArray;
    std::vector<_FLOAT> mRawTriangleAreaArray;

    // ========= stencils ==========
    tEigenArr<tVector4i> mConstraintVertexLst;
    // ========= storage ===========
    std::vector<_FLOAT>
        mBetaArray; // beta is the normal angles (pi - dihedral angles)
    tEigenArr<tVector12> mBetaGradArray;
    tEigenArr<tMatrix12> mBetaHessArray;
    void InitConstraintVertex();
};