#pragma once
#include "utils/DefUtil.h"
#include "utils/EigenUtil.h"
#include <string>
#include <vector>

struct tTriangle;
struct tEdge;
struct tVertex;
SIM_DECLARE_PTR(tTriangle);
SIM_DECLARE_PTR(tEdge);
SIM_DECLARE_PTR(tVertex);
SIM_DECLARE_STRUCT_AND_PTR(tMeshMaterialInfo);
/**
 * \brief           handle everything about obj
 */

class cObjUtil
{
public:
    static void LoadObj(const std::string &path,
                        std::vector<tVertexPtr> &mVertexArray,
                        std::vector<tEdgePtr> &mEdgeArray,
                        std::vector<tTrianglePtr> &mTriangleArray,
                        std::vector<tMeshMaterialInfoPtr> &mMatInfoArray);
    static void
    BuildPlaneGeometryData(const _FLOAT scale, const tVector4 &plane_equation,
                           std::vector<tVertexPtr> &mVertexArray,
                           std::vector<tEdgePtr> &mEdgeArray,
                           std::vector<tTrianglePtr> &mTriangleArray);

    static void BuildEdge(const std::vector<tVertexPtr> &mVertexArray,
                          std::vector<tEdgePtr> &mEdgeArray,
                          const std::vector<tTrianglePtr> &mTriangleArray);
    static bool ExportObj_NoMaterial_WithSimUV(
        std::string export_path, const std::vector<tVertexPtr> &vertices_array,
        const std::vector<tTrianglePtr> &triangles_array, bool silent = false);
    static std::string ExportObj2Str_NoMaterial_WithSimUV(
        const std::vector<tVertexPtr> &vertices_array,
        const std::vector<tTrianglePtr> &triangles_array);

    // output with textures
    static std::string ExportObj2Str_SingleFaceMaterial(
        const std::vector<tVertexPtr> &vertices_array,
        const std::vector<tTrianglePtr> &triangles_array,
        tMeshMaterialInfoPtr single_face_mat);
    static std::string ExportObj2Str_DoubleFaceMaterial(
        const std::vector<tVertexPtr> &vertices_array,
        const std::vector<tTrianglePtr> &triangles_array,
        tMeshMaterialInfoPtr positive_face_mat,
        tMeshMaterialInfoPtr negative_face_mat);
};
