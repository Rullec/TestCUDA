#pragma once
#include "CudaDef.h"
#include "CudaMath.h"
#include <climits>
// #include "SeMath.h"
#define DISABLE_NAN_CHECK

#define ENABLE_CUDA_MEMORY_CHECK

#ifndef ENABLE_CUDA_MEMORY_CHECK

#define CUDA_function

template <typename Type> using devPtr = Type *;

#else

#ifndef __device__
#define __device__
#endif

static __device__ char sg_KernelNameStr[64];
static __device__ char sg_ReadWriteOutOfRange[64];

#define CUDA_function                                                          \
    std::memcpy(sg_KernelNameStr, __FUNCTION__,                                \
                SIM_MIN(sizeof(__FUNCTION__), sizeof(sg_KernelNameStr)));      \
    sg_KernelNameStr[63] = 0;

/**

 */
template <typename Type> class devPtr
{

public:

    SIM_CUDA_CALLABLE operator Type *() { return m_Ptr; }


    SIM_CUDA_CALLABLE operator const Type *() const { return m_Ptr; }


    SIM_CUDA_CALLABLE devPtr(std::nullptr_t) : m_Ptr(nullptr), m_Size(0) {}


    SIM_CUDA_CALLABLE devPtr(Type *_Ptr) : m_Ptr(_Ptr), m_Size(UINT_MAX) {}


    SIM_CUDA_CALLABLE explicit devPtr(Type *_Ptr, unsigned int _Size)
        : m_Ptr(_Ptr), m_Size(_Size)
    {
    }


    SIM_CUDA_CALLABLE operator devPtr<const Type>() const
    {
        return devPtr<const Type>(m_Ptr, m_Size);
    }


    template <typename AnyType> SIM_CUDA_CALLABLE operator AnyType *()
    {
        return reinterpret_cast<AnyType *>(m_Ptr);
    }


    template <typename AnyType>
    SIM_CUDA_CALLABLE operator const AnyType *() const
    {
        return reinterpret_cast<AnyType *>(m_Ptr);
    }


    SIM_CUDA_CALLABLE devPtr<const Type> operator+(unsigned int offset) const
    {
        return devPtr<const Type>(m_Ptr + offset,
                                  (offset < m_Size) ? (m_Size - offset) : 0);
    }


    SIM_CUDA_CALLABLE devPtr<Type> operator+(unsigned int offset)
    {
        return devPtr<Type>(m_Ptr + offset,
                            (offset < m_Size) ? (m_Size - offset) : 0);
    }


    SIM_CUDA_CALLABLE void operator+=(unsigned int offset)
    {
        m_Size = (offset < m_Size) ? (m_Size - offset) : 0;

        m_Ptr += offset;
    }


    SIM_CUDA_CALLABLE const Type &operator[](unsigned int index) const
    {
        if (index >= m_Size)
        {
            if (std::is_same_v<Type, int>)
                printf("[%s]: Device 1D array (int) out of range(%d), "
                       "requested: %d.\n",
                       sg_KernelNameStr, m_Size, index);
            else if (std::is_same_v<Type, float>)
                printf("[%s]: Device 1D array (float) out of range 1 (%d), "
                       "requested: %d.\n",
                       sg_KernelNameStr, m_Size, index);

            return *reinterpret_cast<const Type *>(sg_ReadWriteOutOfRange);
        }
#ifndef DISABLE_NAN_CHECK
        else if (cCudaMath::IsNan(m_Ptr[index]))
        {
            if (std::is_same_v<Type, float>)
                printf("[%s]: Data of device 1D array (float) is not a number, "
                       "index = %d.\n",
                       sg_KernelNameStr, index);
            else
                printf("[%s]: Data of device 1D array (sizeof %lld) is not a "
                       "number, index = %d.\n",
                       sg_KernelNameStr, sizeof(Type), index);
        }
#endif
        return m_Ptr[index];
    }


    SIM_CUDA_CALLABLE Type &operator[](unsigned int index)
    {
        if (index >= m_Size)
        {
            if (std::is_same_v<Type, int>)
                printf("[%s]: Device 1D array (int) out of range(%d), "
                       "requested: %d.\n",
                       sg_KernelNameStr, m_Size, index);
            else if (std::is_same_v<Type, float>)
                printf("[%s]: Device 1D array (float) out of range 2 (%d), "
                       "requested: %d.\n",
                       sg_KernelNameStr, m_Size, index);
            else
                printf("[%s]: Device 1D array (sizeof %lld) out of range(%d), "
                       "requested: %d.\n",
                       sg_KernelNameStr, sizeof(Type), m_Size, index);

            return *reinterpret_cast<Type *>(sg_ReadWriteOutOfRange);
        }

        return m_Ptr[index];
    }

private:
    Type *m_Ptr;

    unsigned int m_Size;
};

/**

 */
template <typename Type> class devPtr<const Type>
{

public:

    SIM_CUDA_CALLABLE operator const Type *() const { return m_Ptr; }


    SIM_CUDA_CALLABLE devPtr(std::nullptr_t) : m_Ptr(nullptr), m_Size(0) {}


    SIM_CUDA_CALLABLE devPtr(const Type *_Ptr) : m_Ptr(_Ptr), m_Size(UINT_MAX)
    {
    }


    SIM_CUDA_CALLABLE explicit devPtr(const Type *_Ptr, unsigned int _Size)
        : m_Ptr(_Ptr), m_Size(_Size)
    {
    }


    template <typename AnyType>
    SIM_CUDA_CALLABLE operator const AnyType *() const
    {
        return reinterpret_cast<const AnyType *>(m_Ptr);
    }


    SIM_CUDA_CALLABLE devPtr<const Type> operator+(unsigned int offset) const
    {
        return devPtr<const Type>(m_Ptr + offset,
                                  (offset < m_Size) ? (m_Size - offset) : 0);
    }


    SIM_CUDA_CALLABLE void operator+=(unsigned int offset)
    {
        m_Size = (offset < m_Size) ? (m_Size - offset) : 0;

        m_Ptr += offset;
    }


    SIM_CUDA_CALLABLE const Type &operator[](unsigned int index) const
    {
        bool exceed = index >= m_Size;
        // printf("[st] index %u size %u, index >= size %u\n", index, m_Size,
        // exceed);
        if (true == exceed)
        {
            if (std::is_same_v<Type, int>)
                printf("[%s]: Device 1D array (int) out of range(%d), "
                       "requested: %d.\n",
                       sg_KernelNameStr, m_Size, index);
            else if (std::is_same_v<Type, float>)
            {
                // printf("[prep] index %u size %u, index >= size %d, exceed =
                // %d\n", index, m_Size, index >= m_Size, exceed);
                printf("[%s]: Device 1D array (float) out of range 3(%d), "
                       "requested: %d.\n",
                       sg_KernelNameStr, m_Size, index);
                if (exceed == false)
                {
                    printf("!!! we should not warn\n");
                }
            }
            else
                printf("[%s]: Device 1D array (sizeof %lld) out of range(%d), "
                       "requested: %d.\n",
                       sg_KernelNameStr, sizeof(Type), m_Size, index);

            return *reinterpret_cast<const Type *>(sg_ReadWriteOutOfRange);
        }
#ifndef DISABLE_NAN_CHECK
        else if (cCudaMath::IsNan(m_Ptr[index]))
        {
            if (std::is_same_v<Type, float>)
                printf("[%s]: Data of device 1D array (float) is not a number, "
                       "index = %d.\n",
                       sg_KernelNameStr, index);
            else
                printf("[%s]: Data of device 1D array (sizeof %lld) is not a "
                       "number, index = %d.\n",
                       sg_KernelNameStr, sizeof(Type), index);
        }
#endif
        return m_Ptr[index];
    }

private:
    const Type *m_Ptr;

    unsigned int m_Size;
};

#endif

/**

 */
template <typename Type> class devPtr2
{

public:

    SIM_CUDA_CALLABLE devPtr2() : m_Ptr(nullptr), m_Rows(0), m_Columns(0) {}


    SIM_CUDA_CALLABLE devPtr2(std::nullptr_t)
        : m_Ptr(nullptr), m_Rows(0), m_Columns(0)
    {
    }


    SIM_CUDA_CALLABLE operator devPtr2<const Type>() const
    {
        return devPtr2<const Type>(m_Ptr, m_Rows, m_Columns);
    }


    SIM_CUDA_CALLABLE explicit devPtr2(Type *_Ptr, unsigned int _Rows,
                                       unsigned int _Columns)
        : m_Ptr(_Ptr), m_Rows(_Rows), m_Columns(_Columns)
    {
    }


    SIM_CUDA_CALLABLE bool IsEmpty() const
    {
        return (m_Ptr == nullptr) || (m_Rows * m_Columns == 0);
    }


    SIM_CUDA_CALLABLE size_t Bytes() const
    {
        return sizeof(Type) * m_Rows * m_Columns;
    }


    SIM_CUDA_CALLABLE unsigned int Size() const { return m_Rows * m_Columns; }


    SIM_CUDA_CALLABLE unsigned int Columns() const { return m_Columns; }


    SIM_CUDA_CALLABLE unsigned int Rows() const { return m_Rows; }

#ifdef ENABLE_CUDA_MEMORY_CHECK


    SIM_CUDA_CALLABLE devPtr<const Type> operator[](unsigned int i) const
    {
        if (i >= static_cast<int>(m_Rows))
        {
            printf("[%s]: Device 2D array (sizeof %lld) out of rows(%d), "
                   "requested: %d.\n",
                   sg_KernelNameStr, sizeof(Type), m_Rows, i);

            return reinterpret_cast<const Type *>(sg_ReadWriteOutOfRange);
        }

        return devPtr<const Type>(m_Ptr + i * m_Columns, m_Columns);
    }


    SIM_CUDA_CALLABLE devPtr<Type> operator[](unsigned int i)
    {
        if (i >= static_cast<int>(m_Rows))
        {
            printf("[%s]: Device 2D array (sizeof %lld) out of rows(%d), "
                   "requested: %d.\n",
                   sg_KernelNameStr, sizeof(Type), m_Rows, i);

            return reinterpret_cast<Type *>(sg_ReadWriteOutOfRange);
        }

        return devPtr<Type>(m_Ptr + i * m_Columns, m_Columns);
    }

    SIM_CUDA_CALLABLE operator devPtr<const Type>() const
    {
        return devPtr<const Type>(m_Ptr, m_Rows * m_Columns);
    }

    SIM_CUDA_CALLABLE operator devPtr<Type>()
    {
        return devPtr<Type>(m_Ptr, m_Rows * m_Columns);
    }
#else

    SIM_CUDA_CALLABLE devPtr<const Type> operator[](unsigned int i) const
    {
        return m_Ptr + i * m_Columns;
    }


    SIM_CUDA_CALLABLE devPtr<Type> operator[](unsigned int i)
    {
        return m_Ptr + i * m_Columns;
    }

    SIM_CUDA_CALLABLE operator devPtr<const Type>() const { return m_Ptr; }

    SIM_CUDA_CALLABLE operator devPtr<Type>() { return m_Ptr; }

#endif

private:
    Type *m_Ptr;

    unsigned int m_Rows;

    unsigned int m_Columns;
};

/**

 */
template <typename Type> class devPtr2<const Type>
{

public:

    SIM_CUDA_CALLABLE devPtr2() : m_Ptr(nullptr), m_Rows(0), m_Columns(0) {}


    SIM_CUDA_CALLABLE devPtr2(std::nullptr_t)
        : m_Ptr(nullptr), m_Rows(0), m_Columns(0)
    {
    }


    SIM_CUDA_CALLABLE explicit devPtr2(const Type *_Ptr, unsigned int _Rows,
                                       unsigned int _Columns)
        : m_Ptr(_Ptr), m_Rows(_Rows), m_Columns(_Columns)
    {
    }


    SIM_CUDA_CALLABLE bool IsEmpty() const
    {
        return (m_Ptr == nullptr) || (m_Rows * m_Columns == 0);
    }


    SIM_CUDA_CALLABLE size_t Bytes() const
    {
        return sizeof(Type) * m_Rows * m_Columns;
    }


    SIM_CUDA_CALLABLE unsigned int Size() const { return m_Rows * m_Columns; }


    SIM_CUDA_CALLABLE unsigned int Columns() const { return m_Columns; }


    SIM_CUDA_CALLABLE unsigned int Rows() const { return m_Rows; }

#ifdef ENABLE_CUDA_MEMORY_CHECK


    SIM_CUDA_CALLABLE devPtr<const Type> operator[](unsigned int i) const
    {
        if (i >= static_cast<int>(m_Rows))
        {
            printf("[%s]: Device 2D array (sizeof %lld) out of rows(%d), "
                   "requested: %d.\n",
                   sg_KernelNameStr, sizeof(Type), m_Rows, i);

            return reinterpret_cast<const Type *>(sg_ReadWriteOutOfRange);
        }

        return devPtr<const Type>(m_Ptr + i * m_Columns, m_Columns);
    }

    SIM_CUDA_CALLABLE operator devPtr<const Type>() const
    {
        return devPtr<const Type>(m_Ptr, m_Rows * m_Columns);
    }
#else

    SIM_CUDA_CALLABLE devPtr<const Type> operator[](unsigned int i) const
    {
        return m_Ptr + i * m_Columns;
    }

    SIM_CUDA_CALLABLE operator devPtr<const Type>() const { return m_Ptr; }

#endif

private:
    const Type *m_Ptr;

    unsigned int m_Rows;

    unsigned int m_Columns;
};