#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_


class cCudaUtil
{
public:
    static size_t GetSharedMemoryBytes(int device_id = 0);
    // static size_t GetRegisterBytes();
};

#endif
