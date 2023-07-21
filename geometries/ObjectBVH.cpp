#include "geometries/ObjectBVH.h"
#include "geometries/Primitives.h"
#include "utils/LogUtil.h"
#include "utils/RenderUtil.h"
#include "utils/TimeUtil.hpp"
#include <iostream>
#include <set>
tBVHNode::tBVHNode()
{
    mId = -1;
    mIsLeaf = false;

    mTriangleId = -1;
    mLeft = nullptr;
    mRight = nullptr;
}
cObjBVH::cObjBVH()
{
    mVertexArray.clear();
    mEdgeArray.clear();
    mTriangleArray.clear();
    mNodes.clear();
}
void printBT(const std::string &prefix, const tBVHNodePtr node, bool isLeft)
{
    if (node != nullptr)
    {
        std::cout << prefix;

        std::cout << (isLeft ? "\u251C\u2500\u2500" : "\u2514\u2500\u2500");

        // print the value of the node
        if (node->mIsLeaf == false)
        {

            std::cout << node->mId << std::endl;
        }
        else
        {
            std::cout << node->mId << " tri " << node->mTriangleId << " mid "
                      << node->mAABB.GetMiddle().segment(0, 3).transpose()
                      << " extent "
                      << node->mAABB.GetExtent().segment(0, 3).transpose()
                      << std::endl;
        }

        // enter the next tree level - left and right branch
        printBT(prefix + (isLeft ? "\u2502  " : "   "), node->mLeft, true);
        printBT(prefix + (isLeft ? "\u2502  " : "   "), node->mRight, false);
    }
}

void printBT(const tBVHNodePtr node) { printBT("", node, false); }
/*
CPU fast BVH construction
*/
void cObjBVH::Init(int obj_id, const std::vector<tVertexPtr> &v_array,
                   const std::vector<tEdgePtr> &e_array,
                   const std::vector<tTrianglePtr> &t_array,
                   _FLOAT outward_half_thickness)
{
    cTimeUtil::Begin("init_bvh");
    mObjId = obj_id;
    mOutwardHalfThickness = outward_half_thickness;
    this->mVertexArray = v_array;
    this->mEdgeArray = e_array;
    this->mTriangleArray = t_array;
    mNodes.clear();
    mLeafNodes.clear();
    RebuildTree();
    printf("[log] init bvh cost %.1f ms\n", cTimeUtil::End("init_bvh", true));
}

void UpdateAABBSubtree(const std::vector<tVertexPtr> &v_array,
                       const std::vector<tEdgePtr> &e_array,
                       const std::vector<tTrianglePtr> &t_array,
                       tBVHNodePtr node, _FLOAT outward_half_thickness)
{
    if (node == nullptr)
        return;
    node->mAABB.Reset();
    if (node->mIsLeaf)
    {

        auto tri = t_array[node->mTriangleId];
        bool is_bdy_tri = e_array[tri->mEId0]->mIsBoundary ||
                          e_array[tri->mEId1]->mIsBoundary ||
                          e_array[tri->mEId2]->mIsBoundary;

        // this node works for a triangle
        // if (is_bdy_tri)
        node->mAABB.ExpandTriangleWithThickness(
            v_array[tri->mId0]->mPos.segment<3>(0),
            v_array[tri->mId1]->mPos.segment<3>(0),
            v_array[tri->mId2]->mPos.segment<3>(0), outward_half_thickness,
            false);
        // else
        // {
        //     node->mAABB.Expand(v_array[tri->mId0]);
        //     node->mAABB.Expand(v_array[tri->mId1]);
        //     node->mAABB.Expand(v_array[tri->mId2]);
        // }

        // increase 1 mm
        // node->mAABB.Increase(tVector3::Ones() * 1e-3);
    }
    else
    {
        UpdateAABBSubtree(v_array, e_array, t_array, node->mLeft,
                          outward_half_thickness);
        UpdateAABBSubtree(v_array, e_array, t_array, node->mRight,
                          outward_half_thickness);
        if (node->mLeft)
            node->mAABB.Expand(node->mLeft->mAABB);
        if (node->mRight)
            node->mAABB.Expand(node->mRight->mAABB);
    }
}
/**
 * \brief           only update the AABB from the leaf node
 */
void cObjBVH::UpdateAABB()
{
    UpdateAABBSubtree(mVertexArray, mEdgeArray, mTriangleArray, mNodes[0],
                      this->mOutwardHalfThickness);
    need_update_rendering_resource = true;
}

template <int N> struct less_than_key
{
    inline bool operator()(const std::pair<int, tVector3> &struct1,
                           const std::pair<int, tVector3> &struct2)
    {
        static_assert(N >= 0 && N <= 3);
        return (struct1.second[N] < struct2.second[N]);
    }
};

/**
 * \brief       create subtree
 * \param       ideal AABB of this node (used for axis split)
 * \param       all vertices included in this node (global vid array)
 * \param       x y z axis vertex sort result       (global vid)
 */
tBVHNodePtr
cObjBVH::CreateSubTree(const tAABB &node_ideal_AABB_used_for_split,
                       const std::vector<int> &vertices_array_in_this_node,
                       const std::vector<int> *local_vertex_id_sorted_xyz)
{
    int num_of_v = vertices_array_in_this_node.size();
    SIM_ASSERT(num_of_v == local_vertex_id_sorted_xyz[0].size());
    SIM_ASSERT(num_of_v == local_vertex_id_sorted_xyz[1].size());
    SIM_ASSERT(num_of_v == local_vertex_id_sorted_xyz[2].size());
    if (num_of_v == 0)
    {
        return nullptr;
    }

    auto node = std::make_shared<tBVHNode>();
    node->mId = mNodes.size();
    node->mAABB = node_ideal_AABB_used_for_split;
    mNodes.push_back(node);

    if (num_of_v == 1)
    {
        // include only one primitive
        node->mIsLeaf = true;
        node->mTriangleId = vertices_array_in_this_node[0];
        node->mLeft = nullptr;
        node->mRight = nullptr;
        mLeafNodes.push_back(node);
    }
    else
    {
        node->mIsLeaf = false;
        node->mTriangleId = -1;
        // include two or more primitve
        // split them into left and right
        int split_dim_id =
            node_ideal_AABB_used_for_split.GetDimIdWithMaxExtent();
        // std::cout << "box = "
        //           << (node_ideal_AABB_used_for_split.mMax -
        //               node_ideal_AABB_used_for_split.mMin)
        //                  .transpose()
        //           << " split dim id = " << split_dim_id << std::endl;
        const std::vector<int> &local_vid_sorted_int_split_dim =
            local_vertex_id_sorted_xyz[split_dim_id];
        SIM_ASSERT(local_vid_sorted_int_split_dim.size() >= 2);
        int left_right_split_pos =
            (local_vid_sorted_int_split_dim.size() + 1) / 2;
        // [0, left_right_split_pos) : left tree
        // [left_right_split_pos, end] : right tree
        // left tree
        {
            std::vector<int> local_vid_sorted_xyz_left[3] = {{}, {}, {}};
            std::vector<int> local_vid_sorted_xyz_right[3] = {{}, {}, {}};
            std::set<int> left_contained_vid = {};
            std::set<int> right_contained_vid = {};
            std::vector<int> left_contained_vid_array = {};
            std::vector<int> right_contained_vid_array = {};
            for (int j = 0; j < left_right_split_pos; j++)
            {
                left_contained_vid.insert(vertices_array_in_this_node[j]);
                left_contained_vid_array.push_back(
                    vertices_array_in_this_node[j]);
            }
            for (int j = left_right_split_pos; j < num_of_v; j++)
            {
                right_contained_vid.insert(vertices_array_in_this_node[j]);
                right_contained_vid_array.push_back(
                    vertices_array_in_this_node[j]);
            }

            for (int i = 0; i < 3; i++)
            {
                for (auto v_id : local_vertex_id_sorted_xyz[i])
                {
                    if (left_contained_vid.find(v_id) !=
                        left_contained_vid.end())
                    {
                        local_vid_sorted_xyz_left[i].push_back(v_id);
                    }
                    else
                    {
                        local_vid_sorted_xyz_right[i].push_back(v_id);
                    }
                }
            }

            {
                tAABB new_AABB = node_ideal_AABB_used_for_split;

                new_AABB.mMax[split_dim_id] =
                    new_AABB.mMin[split_dim_id] +
                    new_AABB.GetExtent()[split_dim_id] / 2;
                node->mLeft = CreateSubTree(new_AABB, left_contained_vid_array,
                                            local_vid_sorted_xyz_left);
            }
            {
                tAABB new_AABB = node_ideal_AABB_used_for_split;

                new_AABB.mMin[split_dim_id] =
                    new_AABB.mMin[split_dim_id] +
                    new_AABB.GetExtent()[split_dim_id] / 2;
                node->mRight =
                    CreateSubTree(new_AABB, right_contained_vid_array,
                                  local_vid_sorted_xyz_right);
            }
        }
    }
    return node;
}
void cObjBVH::RebuildTree()
{
    // 1. calculate AABB for each triangle
    int num_of_tris = mTriangleArray.size();
    tEigenArr<std::pair<int, tVector3>> triangle_centroid(num_of_tris);
    tEigenArr<tAABB> tri_aabb(num_of_tris);
    tAABB global_AABB;
    std::vector<int> tri_centroid_lst[3] = {{}, {}, {}};
    std::vector<int> tri_id_lst = {};
    for (int t = 0; t < num_of_tris; t++)
    {
        tri_id_lst.push_back(t);
        auto tri = mTriangleArray[t];
        tVector3 v0 = mVertexArray[tri->mId0]->mPos.segment<3>(0),
                 v1 = mVertexArray[tri->mId1]->mPos.segment<3>(0),
                 v2 = mVertexArray[tri->mId2]->mPos.segment<3>(0);
        // update AABB
        bool is_bdy_tri = mEdgeArray[tri->mEId0]->mIsBoundary ||
                          mEdgeArray[tri->mEId1]->mIsBoundary ||
                          mEdgeArray[tri->mEId2]->mIsBoundary;

        // if ()
        tri_aabb[t].ExpandTriangleWithThickness(
            v0, v1, v2, this->mOutwardHalfThickness, false);
        // else
        // {
        //     tri_aabb[t].Expand(v0);
        //     tri_aabb[t].Expand(v1);
        //     tri_aabb[t].Expand(v2);
        // }
        global_AABB.Expand(tri_aabb[t]);
        // update centroid
        // 2. create triangle centroid, sort along x axis, y axis, z axis
        triangle_centroid[t] = std::pair<int, tVector3>(t, (v0 + v1 + v2) / 3);
    }
    // std::cout << "global_AABB = " << global_AABB.GetMiddle().transpose()
    //           << " ext = " << global_AABB.GetExtent().transpose() <<
    //           std::endl;
    // x, y, z,axis sort
    // for (int j = 0; j < 3; j++)

    {
        tri_centroid_lst[0].resize(num_of_tris, -1);
        std::sort(triangle_centroid.begin(), triangle_centroid.end(),
                  less_than_key<0>());
        for (int i = 0; i < num_of_tris; i++)
        {
            tri_centroid_lst[0][i] = triangle_centroid[i].first;
        }
    }
    {
        tri_centroid_lst[1].resize(num_of_tris, -1);
        std::sort(triangle_centroid.begin(), triangle_centroid.end(),
                  less_than_key<1>());
        for (int i = 0; i < num_of_tris; i++)
        {
            tri_centroid_lst[1][i] = triangle_centroid[i].first;
        }
    }
    {
        tri_centroid_lst[2].resize(num_of_tris, -1);
        std::sort(triangle_centroid.begin(), triangle_centroid.end(),
                  less_than_key<2>());
        for (int i = 0; i < num_of_tris; i++)
        {
            tri_centroid_lst[2][i] = triangle_centroid[i].first;
        }
    }

    // 2. begin to create the BVH tree, first create root node
    mNodes.clear();
    CreateSubTree(global_AABB, tri_id_lst, tri_centroid_lst);
    need_update_rendering_resource = true;
}
void cObjBVH::Print() const { printBT(mNodes[0]); }

int cObjBVH::GetNumOfLeaves() const { return this->mLeafNodes.size(); }

const std::vector<tBVHNodePtr> cObjBVH::GetLeaves() const
{
    return this->mLeafNodes;
}

void IntersectWithTree(const tBVHNodePtr outer_node, const tBVHNodePtr cur_tree,
                       std::vector<int> possible_collision_v)
{
    if (cur_tree == nullptr)
        return;
    if (false == outer_node->mAABB.Intersect(cur_tree->mAABB))
    {
        return;
    }
    else
    {
        // intersect!
        if (cur_tree->mIsLeaf == true)
        {
            possible_collision_v.push_back(cur_tree->mTriangleId);
        }
        else
        {
            IntersectWithTree(outer_node, cur_tree->mLeft,
                              possible_collision_v);
            IntersectWithTree(outer_node, cur_tree->mRight,
                              possible_collision_v);
        }
    }
}
/**
 * \brief       query!
 */
std::vector<int> cObjBVH::Intersect(tBVHNodePtr outer_node) const
{
    std::vector<int> intersected_vid = {};
    IntersectWithTree(outer_node, mNodes[0], intersected_vid);
    return intersected_vid;
}

const tBVHNodePtr cObjBVH::GetRootNode() const { return mNodes[0]; }

// Recursive helper function to generate points for the specified layer
// void GeneratePointsForLayer(int layer_id, tBVHNodePtr node,
//                             std::vector<std::vector<tVector3>> &points,
//                             int current_layer)
// {
//     if (node == nullptr)
//         return;

//     // If the current node is at the desired layer, generate points for its
//     AABB if (current_layer == layer_id && !node->mIsLeaf)
//     {

//         points.emplace_back(points_local);
//     }

//     // Recurse for child nodes
//     GeneratePointsForLayer(layer_id, node->mLeft, points, current_layer + 1);
//     GeneratePointsForLayer(layer_id, node->mRight, points, current_layer +
//     1);
// }

// Function to generate points for OpenGL VAO
// std::vector<std::vector<tVector3>>
// cObjBVH::GenerateLinesForEachBoxPerLayer(int layer_id)
// {
//     std::vector<std::vector<tVector3>> points;

//     // Retrieve the root node of the BVH tree
//     tBVHNodePtr root = GetRootNode();

//     // Call a recursive function to traverse the BVH tree and generate points
//     // for the specified layer
//     GeneratePointsForLayer(layer_id, root, points, 0);

//     return points;
// }
#include "utils/ColorUtil.h"
#include <queue>
void cObjBVH::UpdateRenderingResource()
{
    if (need_update_rendering_resource == false)
    {
        return;
    }
    need_update_rendering_resource = false;
    // int nl = this->GetNumOfLayers();
    // this->mRenderBufferBoxesPerLayer.resize(nl);
    // for each layer, get the nodes, create the buffers.
    auto root_node = this->GetRootNode();
    if (root_node == nullptr)
    {
        return;
    }
    this->mRenderBufferBoxesPerLayer.clear();
    // non empty tree
    std::queue<tBVHNodePtr> node_queue;
    node_queue.push(root_node);
    int layer_id = 0;
    while (node_queue.empty() == false)
    {
        int size = node_queue.size();
        tEigenArr<tVector3> points_buf_in_this_layer = {};
        for (int i = 0; i < size; i++)
        {
            auto cur_node = node_queue.front();
            if (cur_node->mLeft != nullptr)
                node_queue.push(cur_node->mLeft);

            if (cur_node->mRight != nullptr)
                node_queue.push(cur_node->mRight);

            const auto &local_pts = cur_node->mAABB.GetAABBPointsForRendering();
            points_buf_in_this_layer.insert(points_buf_in_this_layer.end(),
                                            local_pts.begin(), local_pts.end());
            node_queue.pop();
        }

        tVectorXf render_buf(points_buf_in_this_layer.size() *
                             RENDERING_SIZE_PER_VERTICE);
        render_buf.setConstant(std::nan(""));

        Eigen::Map<tVectorXf> map(render_buf.data(), render_buf.size());
        int st = 0;
        for (int eid = 0; eid < points_buf_in_this_layer.size() / 2; eid++)
        {
            cRenderUtil::CalcEdgeDrawBufferSingle3(
                points_buf_in_this_layer[2 * eid + 0],
                points_buf_in_this_layer[2 * eid + 1], tVector3::Zero(), map,
                st, ColorBlack);
        }
        mRenderBufferBoxesPerLayer.emplace_back(render_buf);
        layer_id++;
        // SIM_INFO("generate render buffer size {} for BVH layer {}",
        //          points_buf_in_this_layer.size(), layer_id++);
    }

    // begin to create render resource
    this->mRenderResourceArray.clear();
    for (auto &x : mRenderBufferBoxesPerLayer)
    {
        auto resource = std::make_shared<cRenderResource>();
        resource->mName = "bvh";
        resource->mLineBuffer.mNumOfEle = x.size();
        resource->mLineBuffer.mBuffer = x.data();
        mRenderResourceArray.push_back(resource);
    }
}
std::vector<cRenderResourcePtr>
cObjBVH::GetRenderingResource(int layer_id) const
{
    if (layer_id == -1)
    {
        return mRenderResourceArray;
    }
    else if (layer_id >= 0 && layer_id < mRenderResourceArray.size())
    {
        return {mRenderResourceArray[layer_id]};
    }
    else
    {
        return {};
    }
}
#include <queue>
int cObjBVH::GetNumOfLayers() const
{
    auto root_node = this->GetRootNode();
    if (root_node == nullptr)
    {
        return 0;
    }
    int cur_layers = 0;
    std::queue<tBVHNodePtr> node_queue;
    node_queue.push(root_node);
    while (node_queue.empty() == false)
    {
        int size = node_queue.size();
        for (int i = 0; i < size; i++)
        {
            auto cur_node = node_queue.front();
            if (cur_node->mLeft != nullptr)
                node_queue.push(cur_node->mLeft);

            if (cur_node->mRight != nullptr)
                node_queue.push(cur_node->mRight);

            node_queue.pop();
        }
        cur_layers += 1;
    }
    return cur_layers;
}