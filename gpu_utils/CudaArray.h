#ifndef CUDA_ARRAY_H_
#pragma once
#include "CudaAsync.h"
#include "CudaDevPtr.h"
#include "CudaMath.h"
#include "CudaMemory.h"
/**
 */
template <typename Type> class cCudaArray
{
public:
    cCudaArray(){};
    ~cCudaArray(){};
    using size_type = size_t;

    void Resize(size_type Count) { mMemory.Allocate(Count * sizeof(Type)); }

    void Download(Type *pHost, size_type Count) const
    {
        mMemory.ReadFromDevice(pHost, Count * sizeof(Type), 0);
    }

    void Download(std::vector<Type> &HostVec) const
    {
        size_t num_of_ele = this->Size();
        HostVec.resize(num_of_ele);

        mMemory.ReadFromDevice(HostVec.data(), this->Size() * sizeof(Type), 0);
    }

    void Upload(const Type *pHost, size_type Count)
    {
        if (this->Size() < Count)
        {
            this->Resize(Count);
#ifdef ENABLE_CUDA_MEMORY_CHECK
            printf("Device Memory Not Yet Allocated!\n");
#endif
        }

#ifdef ENABLE_CUDA_MEMORY_CHECK
        for (size_type i = 0; i < Count; i++)
        {
            if (cCudaMath::IsNan(pHost[i]))
            {
                printf("Uploading invalid value!");

                break;
            }
        }
#endif

        mMemory.WriteToDevice(pHost, Count * sizeof(Type), 0);
    }

    void Upload(const std::vector<Type> &HostVec)
    {
        size_t bytes_of_type = sizeof(Type);
        size_t size_stl_vec = HostVec.size();
        size_t total_bytes = bytes_of_type * size_stl_vec;
        mMemory.Allocate(total_bytes);

        this->Upload(HostVec.data(), size_stl_vec);
    }

    void MemsetAsync(Type Value)
    {
        if (!this->IsEmpty())
        {
            CudaAsync::Memset(this->Ptr(), Value, this->Size());
        }
    }

    size_type Size() const
    {
        return static_cast<size_type>(mMemory.Bytes()) / sizeof(Type);
    }

    size_type Bytes() const { return static_cast<size_type>(mMemory.Bytes()); }

    bool IsEmpty() const { return mMemory.IsEmpty(); }

    void Clear() { mMemory.Free(); }

#ifdef ENABLE_CUDA_MEMORY_CHECK

    devPtr<const Type> Ptr() const
    {
        // static_assert(false);
        return devPtr<const Type>((const Type *)mMemory.Ptr(), this->Size());
    }

    devPtr<Type> Ptr()
    {
        // static_assert(false);
        return devPtr<Type>((Type *)mMemory.Ptr(), this->Size());
    }

#else

    devPtr<const Type> Ptr() const { return devPtr<const Type>(mMemory.Ptr()); }

    devPtr<Type> Ptr() { return devPtr<Type>(mMemory.Ptr()); }
#endif

protected:
    cCudaMemory mMemory;
};
#endif