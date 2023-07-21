#pragma once
#include "utils/DefUtil.h"
#include "utils/MathUtil.h"

SIM_DECLARE_STRUCT_AND_PTR(tVertex);

struct tAABB
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    explicit tAABB();
    tAABB(const tAABB &old_AABB);
    int GetDimIdWithMaxExtent() const;
    tVector3 GetExtent() const;
    tVector3 GetMiddle() const;
    void Reset();
    void Expand(const tVector3 &);
    void ExpandTriangleWithThickness(const tVector3 &v0, const tVector3 &v1,
                                     const tVector3 &v2, _FLOAT thickness,
                                     bool expand_shape_direction);
    void Expand(const tVector4 &);
    void Expand(const tVertexPtr &);
    void Increase(const tVector3 &dist);
    void Expand(const tAABB &);
    bool IsInvalid() const;
    bool Intersect(const tAABB &other_AABB);
    tEigenArr<tVector3> GetAABBPointsForRendering();
    tVector3 mMin, mMax;
};