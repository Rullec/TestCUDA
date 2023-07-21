set(cuda_src 
CudaArray.cpp
CudaMath.cpp  
CudaELLMatrix.cpp  
CudaMemory.cpp
CudaAsync.cu  
GPUMatrixOps.cu
CublasUtil.cpp
)
add_library(cuda_lib ${cuda_src})

target_compile_options(cuda_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)
target_link_libraries(cuda_lib PUBLIC utils_lib)
# get_property(dirs TARGET cuda_lib PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${dirs})
#   message(STATUS "dir='${dir}'")
# endforeach()