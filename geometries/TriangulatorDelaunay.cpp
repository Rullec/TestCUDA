#include "Triangulator.h"
#include "geometries/Primitives.h"
#include "utils/ColorUtil.h"
#include "utils/LogUtil.h"
#include <delaunator.hpp>

// void cTriangulator::DelaunayTriangulation(
//     int target_num_of_vertices, std::vector<tVertexPtr> vertex_array,
//     std::vector<tEdgePtr> edge_array, std::vector<tTrianglePtr> tri_array)

static void add_edge_into_edge_array(std::vector<tEdgePtr> &edge_array,
                                     int contained_triangle_id, int v0, int v1)
{
    if (v0 > v1)
        SIM_SWAP(v0, v1);
    if (v0 == v1)
    {
        printf("[error] edge loop!");
        exit(1);
    }
    tEdgePtr cur_edge = nullptr;
    for (auto &edge : edge_array)
    {
        if (edge->mId0 == v0 && edge->mId1 == v1)
        {
            cur_edge = edge;
            break;
        }
    }
    if (cur_edge == nullptr)
    {
        // did not find it
        cur_edge = std::make_shared<tEdge>();
        cur_edge->mId0 = v0;
        cur_edge->mId1 = v1;
        cur_edge->mTriangleId0 = contained_triangle_id;
        edge_array.push_back(cur_edge);
    }
    else
    {
        SIM_ASSERT(cur_edge->mTriangleId0 != -1);
        SIM_ASSERT(cur_edge->mTriangleId1 == -1);
        cur_edge->mTriangleId1 = contained_triangle_id;
    }
}
void cTriangulator::DelaunayTriangulation(_FLOAT cloth_width,
                                          _FLOAT cloth_height,
                                          int target_num_of_vertices,
                                          std::vector<tVertexPtr> &v_array,
                                          std::vector<tEdgePtr> &e_array,
                                          std::vector<tTrianglePtr> &tri_array)
{
    // // clear all
    // v_array.clear();
    // e_array.clear();
    // tri_array.clear();

    // // 1. build boundary

    // tEigenArr<tVector2> v_lst = BuildRectangleBoundary(
    //     cloth_width, cloth_height, target_num_of_vertices);

    // FLOAT min_dist_threshold =
    //     std::sqrt((cloth_width * cloth_height / target_num_of_vertices) /
    //     M_PI);
    // while (v_lst.size() < target_num_of_vertices)
    // {
    //     // 1. generate new point
    //     tVector2 new_point = tVector2::Zero();
    //     new_point[0] = cMathUtil::RandFloat(-cloth_width / 2, cloth_width /
    //     2); new_point[1] =
    //         cMathUtil::RandFloat(-cloth_height / 2, cloth_height / 2);
    //     // 2. calculate min distance with others
    //     FLOAT min_distance_with_others = std::numeric_limits<FLOAT>::max();
    //     for (auto &v : v_lst)
    //     {
    //         FLOAT cur_dist = (v - new_point).norm();
    //         min_distance_with_others =
    //             SIM_MIN(min_distance_with_others, cur_dist);
    //     }
    //     // if it is bigger than 5mm, we admit it
    //     if (min_distance_with_others > min_dist_threshold)
    //         v_lst.push_back(new_point);
    // }

    // // 2. collect all vertices
    // std::vector<FLOAT> coord_lst = {};
    // for (int i = 0; i < target_num_of_vertices; i++)
    // {
    //     const tVector2 &cur_v2d = v_lst[i];
    //     tVertexPtr new_v = std::make_shared<tVertex>();
    //     new_v->muv_simple[0] = cur_v2d[0];
    //     new_v->muv_simple[1] = cur_v2d[1];
    //     new_v->mPos[0] = cur_v2d[0];
    //     new_v->mPos[1] = cur_v2d[1];
    //     new_v->mPos[2] = 0;
    //     new_v->mColor = ColorAn;
    //     new_v->mMass = 0;
    //     new_v->mNormal = tVector4(0, 0, 1, 0);
    //     v_array.push_back(new_v);
    //     coord_lst.push_back(cur_v2d[0]);
    //     coord_lst.push_back(cur_v2d[1]);
    // }

    // // 3. triangulation, build triangle list and edge list
    // delaunator::Delaunator d(coord_lst);
    // typedef std::pair<int, int> tEdgeData;

    // auto calc_normal = [](const tVector4 &v0, const tVector4 &v1,
    //                       const tVector4 &v2) -> tVector4
    // { return ((v1 - v0).cross3(v2 - v1)).normalized(); };

    // for (size_t i = 0; i < d.triangles.size(); i += 3)
    // {
    //     int v0 = d.triangles[i], v1 = d.triangles[i + 1],
    //         v2 = d.triangles[i + 2];
    //     // check clockwise or counter-clockwise
    //     if (std::fabs(calc_normal(v_array[v0]->mPos, v_array[v1]->mPos,
    //                               v_array[v2]->mPos)[1] -
    //                   1) > 1e-3)
    //     {
    //         SIM_SWAP(v0, v2);
    //     }
    //     // now th triangle is counter clockwise

    //     auto tri = std::make_shared<tTriangle>();
    //     tri->mId0 = v0;
    //     tri->mId1 = v1;
    //     tri->mId2 = v2;
    //     tri->mNormal = (v_array[v0]->mNormal + v_array[v1]->mNormal +
    //                     v_array[v2]->mNormal) /
    //                    3;

    //     // add triangle
    //     tri_array.push_back(tri);

    //     // add edge
    //     add_edge_into_edge_array(e_array, tri_array.size() - 1, v0, v1);
    //     add_edge_into_edge_array(e_array, tri_array.size() - 1, v1, v2);
    //     add_edge_into_edge_array(e_array, tri_array.size() - 1, v2, v0);
    // }

    // // 4. set other edge result
    // for (int i = 0; i < e_array.size(); i++)
    // {
    //     auto &e = e_array[i];
    //     if (e->mTriangleId0 != -1 && e->mTriangleId1 == -1)
    //         e->mIsBoundary = true;
    //     else if (e->mTriangleId0 != -1 && e->mTriangleId1 != -1)
    //         e->mIsBoundary = false;
    //     else
    //     {
    //         printf("[error] illegal case! e triangle0 %d 1 %d\n",
    //                e->mTriangleId0, e->mTriangleId1);
    //         exit(1);
    //     }
    //     // std::cout << "e " << i << " is boundary " << e->mIsBoundary << "
    //     tri0 "
    //     //           << e->mTriangleId0 << " tri1 " << e->mTriangleId1
    //     //           << std::endl;
    //     e->mRawLength =
    //         (v_array[e->mId0]->mPos - v_array[e->mId1]->mPos).norm();
    //     e->mK_spring = 0;
    //     SIM_ASSERT(e->mTriangleId0 < static_cast<int>(tri_array.size()));
    //     SIM_ASSERT(e->mTriangleId1 < static_cast<int>(tri_array.size()));
    // }

    // // cTriangulator::ValidateGeometry(v_array, e_array, tri_array);
    // printf("[log] delaunay triangulation done, num_of_v %d, num_of_e %d, "
    //        "num_of_tri %d\n",
    //        v_array.size(), e_array.size(), tri_array.size());

    // judge: edge raw length = texture coords
    // for (int i = 0; i < e_array.size(); i++)
    // {
    //     auto cur_e = e_array[i];

    //     auto v0 = v_array[cur_e->mId0], v1 = v_array[cur_e->mId1];
    //     FLOAT uv_dist = (v0->muv_simple - v1->muv_simple).norm(),
    //           cartesian_dist = (v0->mPos - v1->mPos).norm();
    //     FLOAT diff = std::fabs(uv_dist - cartesian_dist);
    //     std::cout << "edge " << i << " v" << cur_e->mId0 << " to v"
    //               << cur_e->mId1 << " uv dist = " << uv_dist
    //               << " car dist = " << cartesian_dist << " diff = " << diff
    //               << std::endl;
    // }
}

tEigenArr<tVector2> cTriangulator::BuildRectangleBoundary(_FLOAT width,
                                                          _FLOAT height,
                                                          int num_of_vertices)
{
    _FLOAT density = std::sqrt(height * width / num_of_vertices); // m per point
    int num_of_point_x_axis =
            SIM_MAX(static_cast<int>(std::round(width / density)), 2),
        num_of_point_y_axis =
            SIM_MAX(static_cast<int>(std::round(height / density)), 2);
    _FLOAT point_gap_x_axis = width / (num_of_point_x_axis - 1),
           point_gap_y_axis = height / (num_of_point_y_axis - 1);
    _FLOAT left = -width / 2, right = width / 2, up = height / 2,
           down = -height / 2;
    tEigenArr<tVector2> array = {};

    // bottom left -> bottom right
    for (int i = 1; i < num_of_point_x_axis; i++)
        array.push_back(tVector2(left + i * point_gap_x_axis, down));

    // bottom right -> up right
    for (int i = 1; i < num_of_point_y_axis; i++)
        array.push_back(tVector2(right, down + i * point_gap_y_axis));

    // up right -> up left
    for (int i = num_of_point_x_axis - 2; i >= 0; i--)
        array.push_back(tVector2(left + i * point_gap_x_axis, up));

    // up left -> bottom left
    for (int i = num_of_point_y_axis - 2; i >= 0; i--)
        array.push_back(tVector2(left, down + i * point_gap_y_axis));
    std::cout << "array size = " << array.size()
              << " target num = " << num_of_vertices << std::endl;
    return array;
}

void cTriangulator::CalcAABB(const std::vector<tVertexPtr> &v_array,
                             tVector4 &aabb_min, tVector4 &aabb_max)
{

    aabb_min = tVector4::Ones() * std::numeric_limits<float>::max();
    aabb_max = tVector4::Ones() * std::numeric_limits<float>::max() * -1;
    for (auto &x : v_array)
    {
        for (int i = 0; i < 3; i++)
        {

            _FLOAT val = x->mPos[i];
            aabb_min[i] = (val < aabb_min[i]) ? val : aabb_min[i];
            aabb_max[i] = (val > aabb_max[i]) ? val : aabb_max[i];
        }
    }
}
void cTriangulator::MoveToOrigin(std::vector<tVertexPtr> &v_array)
{
    tVector4 aabb_min, aabb_max;
    CalcAABB(v_array, aabb_min, aabb_max);
    tVector4 center = (aabb_max + aabb_min) / 2;
    for(auto & x : v_array)
    {
        x->mPos.segment<3>(0) -= center.segment<3>(0);
    }
}