
add_library(
    geo_lib
    ObjExport.cpp
    Primitives.cpp
    Triangulator.cpp
    ObjUtil.cpp
    hermite/Hermite2D.cpp
    hermite/Hermite1D.cpp
    # Raycaster.cpp
    BezierCurve.cpp
    Tetrahedron.cpp
    TriangulatorDelaunay.cpp
    DihedralAngles.cpp 
    DihedralAnglesMachine.cpp 
    # FundamentalForms.cpp
    NormalEdgeCross.cpp
    ObjectBVH.cpp
    AABB.cpp
)

target_link_libraries(geo_lib PUBLIC utils_lib)

# message(FATAL_ERROR ${PROJECT_SOURCE_DIR})

install(TARGETS geo_lib RUNTIME DESTINATION bin)
message("install " geo_lib " to " ${PROJECT_SOURCE_DIR})