#include "AABB.h"
#include "geometries/Primitives.h"
#include "utils/LogUtil.h"
tAABB::tAABB() { Reset(); }

void tAABB::Expand(const tVector4 &vec4)
{
    tVector3 val = vec4.segment<3>(0);
    Expand(val);
}
void tAABB::Expand(const tVector3 &vec)
{
    if (IsInvalid() == true)
    {
        mMin = vec;
        mMax = vec;
    }
    else
    {
        for (int i = 0; i < 3; i++)
        {
            mMin[i] = std::min(vec[i], mMin[i]);
            mMax[i] = std::max(vec[i], mMax[i]);
        }
    }
}
void tAABB::Expand(const tVertexPtr &ptr) { Expand(ptr->mPos); }

bool tAABB::IsInvalid() const
{
    if (true == (mMin.hasNaN() || mMax.hasNaN()))
    {
        return true;
    }
    else
    {
        SIM_ASSERT((mMax - mMin).minCoeff() >= 0);
        return false;
    }
}

void tAABB::Expand(const tAABB &new_AABB)
{
    if (new_AABB.IsInvalid())
    {
        // SIM_WARN("new AABB is invalid!");
        return;
    }
    if (IsInvalid() == true)
    {
        mMin = new_AABB.mMin;
        mMax = new_AABB.mMax;
    }
    else
    {

        for (int i = 0; i < 3; i++)
        {
            mMin[i] = std::min(new_AABB.mMin[i], mMin[i]);
            mMax[i] = std::max(new_AABB.mMax[i], mMax[i]);
        }
    }
}

int tAABB::GetDimIdWithMaxExtent() const
{
    tVector3 extent = mMax - mMin;
    float max_extent = extent[0];
    int max_extent_id = 0;
    for (int i = 1; i < 3; i++)
    {
        if (extent[i] > max_extent)
        {
            max_extent = extent[i];
            max_extent_id = i;
        }
    }
    return max_extent_id;
}

tAABB::tAABB(const tAABB &old_AABB)
{
    mMin = old_AABB.mMin;
    mMax = old_AABB.mMax;
}
tVector3 tAABB::GetExtent() const { return mMax - mMin; }
tVector3 tAABB::GetMiddle() const
{
    tVector3 middle = (mMax + mMin) / 2;
    return middle;
}

bool tAABB::Intersect(const tAABB &other_AABB)
{
    return ((mMin[0] <= other_AABB.mMax[0]) &&
            (mMax[0] >= other_AABB.mMin[0])) &&
           ((mMin[1] <= other_AABB.mMax[1]) &&
            (mMax[1] >= other_AABB.mMin[1])) &&
           ((mMin[2] <= other_AABB.mMax[2]) && (mMax[2] >= other_AABB.mMin[2]));
}
void tAABB::Reset()
{
    mMin = std::nan("") * tVector3::Ones();
    mMax = std::nan("") * tVector3::Ones();
}
void tAABB::Increase(const tVector3 &dist)
{
    mMin -= dist;
    mMax += dist;
}

tEigenArr<tVector3> tAABB::GetAABBPointsForRendering()
{
    // Generate the eight corner points of the AABB
    tEigenArr<tVector3> points_local = {};
    // points_local.emplace_back(mMin.x(), mMin.y(), mMin.z());
    // points_local.emplace_back(mMax.x(), mMin.y(), mMin.z());
    // points_local.emplace_back(mMax.x(), mMax.y(), mMin.z());
    // points_local.emplace_back(mMin.x(), mMax.y(), mMin.z());
    // points_local.emplace_back(mMin.x(), mMin.y(), mMax.z());
    // points_local.emplace_back(mMax.x(), mMin.y(), mMax.z());
    // points_local.emplace_back(mMax.x(), mMax.y(), mMax.z());
    // points_local.emplace_back(mMin.x(), mMax.y(), mMax.z());
    points_local.emplace_back(mMin.x(), mMin.y(), mMin.z());
    points_local.emplace_back(mMax.x(), mMin.y(), mMin.z());

    points_local.emplace_back(mMax.x(), mMin.y(), mMin.z());
    points_local.emplace_back(mMax.x(), mMax.y(), mMin.z());

    points_local.emplace_back(mMax.x(), mMax.y(), mMin.z());
    points_local.emplace_back(mMin.x(), mMax.y(), mMin.z());

    points_local.emplace_back(mMin.x(), mMax.y(), mMin.z());
    points_local.emplace_back(mMin.x(), mMin.y(), mMin.z());

    points_local.emplace_back(mMin.x(), mMin.y(), mMax.z());
    points_local.emplace_back(mMax.x(), mMin.y(), mMax.z());

    points_local.emplace_back(mMax.x(), mMin.y(), mMax.z());
    points_local.emplace_back(mMax.x(), mMax.y(), mMax.z());

    points_local.emplace_back(mMax.x(), mMax.y(), mMax.z());
    points_local.emplace_back(mMin.x(), mMax.y(), mMax.z());

    points_local.emplace_back(mMin.x(), mMax.y(), mMax.z());
    points_local.emplace_back(mMin.x(), mMin.y(), mMax.z());

    points_local.emplace_back(mMin.x(), mMin.y(), mMin.z());
    points_local.emplace_back(mMin.x(), mMin.y(), mMax.z());

    points_local.emplace_back(mMax.x(), mMin.y(), mMin.z());
    points_local.emplace_back(mMax.x(), mMin.y(), mMax.z());

    points_local.emplace_back(mMax.x(), mMax.y(), mMin.z());
    points_local.emplace_back(mMax.x(), mMax.y(), mMax.z());

    points_local.emplace_back(mMin.x(), mMax.y(), mMin.z());
    points_local.emplace_back(mMin.x(), mMax.y(), mMax.z());

    return points_local;
}

void tAABB::ExpandTriangleWithThickness(const tVector3 &v0, const tVector3 &v1,
                                        const tVector3 &v2,
                                        _FLOAT half_thickness,
                                        bool expand_shape_direction)
{
    tEigenArr<tVector3> newv;

    if (expand_shape_direction == true)
        newv = cMathUtil::ExpandTriangle(v0, v1, v2, half_thickness);
    else
    {
        newv = {v0, v1, v2};
    }
    tVector3 normal =
        ((newv[1] - newv[0]).cross(newv[2] - newv[1])).normalized();
    Expand(tVector3(newv[0] + normal * half_thickness));
    Expand(tVector3(newv[1] + normal * half_thickness));
    Expand(tVector3(newv[2] + normal * half_thickness));
    Expand(tVector3(newv[0] - normal * half_thickness));
    Expand(tVector3(newv[1] - normal * half_thickness));
    Expand(tVector3(newv[2] - normal * half_thickness));
}