#include "CudaIntrinsic.h"
__device__ void AtomicAddMat3(tCudaMatrix3f *address, tCudaMatrix3f value)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            atomicAdd(&(*address)(i, j), value(i, j));
}
__device__ void AtomicAddFloat(float *address, float value)
{
    atomicAdd(address, value);
}
__device__ void AtomicAddVec3(tCudaVector3f *address, tCudaVector3f value)
{
    atomicAdd(&((*address)[0]), value[0]);
    atomicAdd(&((*address)[1]), value[1]);
    atomicAdd(&((*address)[2]), value[2]);
}