#include "utils/TensorUtil.h"
#if EIGEN_MAJOR_VERSION < 4
#error "Please use Eigen 3.4.0+ to support slice"
#endif

namespace nTensorUtil
{

template <int N>
Eigen::array<long, N> CreateTensorIndex(const std::initializer_list<long> &arr)
{
    Eigen::array<long, N> result;
    std::copy(arr.begin(), arr.end(), result.begin());
    return result;
}

template Eigen::array<long, 3>
CreateTensorIndex<3>(const std::initializer_list<long> &arr);
template Eigen::array<long, 4>
CreateTensorIndex<4>(const std::initializer_list<long> &arr);

template <typename dtype, int N>
tVectorXi GetTensorDim(const Eigen::Tensor<dtype, N> &tensor)
{
    tVectorXi dims(tensor.dimensions().size());
    for (int i = 0; i < tensor.dimensions().size(); i++)
    {
        dims[i] = tensor.dimension(i);
    }
    return dims;
}

#define INSTANTIATE_GET_TENSOR_DIM(dtype, id)                                  \
    template tVectorXi GetTensorDim<dtype, id>(                                \
        const Eigen::Tensor<dtype, id> &tensor);

INSTANTIATE_GET_TENSOR_DIM(_FLOAT, 1);
INSTANTIATE_GET_TENSOR_DIM(_FLOAT, 2);
INSTANTIATE_GET_TENSOR_DIM(_FLOAT, 3);
INSTANTIATE_GET_TENSOR_DIM(_FLOAT, 4);
INSTANTIATE_GET_TENSOR_DIM(_FLOAT, 5);
INSTANTIATE_GET_TENSOR_DIM(int, 1);
INSTANTIATE_GET_TENSOR_DIM(int, 2);
INSTANTIATE_GET_TENSOR_DIM(int, 3);
INSTANTIATE_GET_TENSOR_DIM(int, 4);
INSTANTIATE_GET_TENSOR_DIM(int, 5);

template <typename T, int NumDims, int NumNewDims>
Eigen::Tensor<T, NumDims + NumNewDims>
TensorExpand(const Eigen::Tensor<T, NumDims> &input_tensor,
             const std::vector<int> &positions)
{

    //   "The number of new dimensions should match the size of the "
    //   "positions vector."
    assert(NumNewDims == positions.size());

    // Create an array containing the new dimensions
    Eigen::array<Eigen::DenseIndex, NumDims + NumNewDims> new_dims;
    int old_dim_index = 0;
    int new_dim_index = 0;

    for (int i = 0; i < NumDims + NumNewDims; ++i)
    {
        if (new_dim_index < NumNewDims && i == positions[new_dim_index])
        {
            new_dims[i] = 1;
            new_dim_index++;
        }
        else
        {
            new_dims[i] = input_tensor.dimension(old_dim_index);
            old_dim_index++;
        }
    }

    // Reshape the input tensor to the new dimensions
    Eigen::Tensor<T, NumDims + NumNewDims> new_tensor =
        input_tensor.reshape(new_dims);

    return new_tensor;
}

#define INSTANTIATE_ADD_SINGLETON_DIM(dtype, N, M)                             \
    template Eigen::Tensor<dtype, N + M> TensorExpand<dtype, N, M>(            \
        const Eigen::Tensor<dtype, N> &input_tensor,                           \
        const std::vector<int> &positions);

INSTANTIATE_ADD_SINGLETON_DIM(_FLOAT, 1, 1);
INSTANTIATE_ADD_SINGLETON_DIM(_FLOAT, 1, 2);
INSTANTIATE_ADD_SINGLETON_DIM(_FLOAT, 2, 1);
INSTANTIATE_ADD_SINGLETON_DIM(_FLOAT, 2, 2);
INSTANTIATE_ADD_SINGLETON_DIM(_FLOAT, 3, 1);
INSTANTIATE_ADD_SINGLETON_DIM(_FLOAT, 3, 2);
INSTANTIATE_ADD_SINGLETON_DIM(_FLOAT, 4, 1);
INSTANTIATE_ADD_SINGLETON_DIM(_FLOAT, 4, 2);

Eigen::Tensor<_FLOAT, 1> ConvertVectorToTensor(const tVectorX &vec)
{
    Eigen::Tensor<_FLOAT, 1> tensor =
        Eigen::TensorMap<const Eigen::Tensor<_FLOAT, 1>>(vec.data(),
                                                         vec.size());

    return tensor;
}

template <typename T, int NumDims>
Eigen::Tensor<T, NumDims>
expand_tensor(const Eigen::Tensor<T, NumDims> &input_tensor,
              const Eigen::array<int, NumDims> &expansion_factors)
{
    // Create an array containing the expanded dimensions
    Eigen::array<Eigen::DenseIndex, NumDims> expanded_dims;
    for (int i = 0; i < NumDims; ++i)
    {
        expanded_dims[i] = input_tensor.dimension(i) * expansion_factors[i];
    }

    // Broadcast the input tensor to the expanded dimensions
    Eigen::Tensor<T, NumDims> expanded_tensor =
        input_tensor.broadcast(expanded_dims);

    return expanded_tensor;
}

Eigen::Tensor<_FLOAT, 2> ConvertMatrixToTensor(const tMatrixX &mat)
{
    int a = mat.rows(), b = mat.cols();

    const Eigen::TensorMap<const Eigen::Tensor<_FLOAT, 2>> tensor(mat.data(), a,
                                                                  b);
    return tensor;
}

tMatrixX ConvertTensorToMatrix(const tTensor2d &t)
{
    tVectorXi dims = nTensorUtil::GetTensorDim(t);
    Eigen::Map<const tMatrixX> ret_mat(t.data(), dims[0], dims[1]);
    return ret_mat;
}
tVectorX ConvertTensorToVector(const tTensor1d &t)
{
    Eigen::Map<const tVectorX> ret_vec(t.data(), t.dimension(0));
    return ret_vec;
}
// template Eigen::TensorMap<Eigen::Tensor<_FLOAT, 2>>
// ConvertMatrixToTensor<_FLOAT>(
//     const Eigen::Matrix<_FLOAT, Eigen::Dynamic, Eigen::Dynamic> &);

template <int N>
void SetTensorSliceByTensor(
    Eigen::Tensor<_FLOAT, N> &tensor0, const Eigen::Tensor<_FLOAT, N> &tensor1,
    const std::initializer_list<Eigen::Index> &start_idx)
{

    // Check if the dimensions match
    if (tensor0.dimensions().size() != N || tensor1.dimensions().size() != N)
    {
        throw std::runtime_error("The number of dimensions of the tensors does "
                                 "not match the template parameter N.");
    }

    // Check if the number of start indices matches the number of dimensions
    if (start_idx.size() != N)
    {
        throw std::runtime_error("The number of start indices does not match "
                                 "the number of dimensions.");
    }

    // Create the 'start_indices' array from the initializer_list
    Eigen::array<Eigen::Index, N> start_indices;
    std::copy(start_idx.begin(), start_idx.end(), start_indices.begin());

    // Create the 'extents' array from the dimensions of tensor1
    Eigen::array<Eigen::Index, N> extents;
    for (int i = 0; i < N; ++i)
    {
        extents[i] = tensor1.dimension(i);
    }

    // Assign the sliced part of tensor0 to tensor1
    tensor0.slice(start_indices, extents) = tensor1;
}

template void
SetTensorSliceByTensor<3>(Eigen::Tensor<_FLOAT, 3> &tensor0,
                          const Eigen::Tensor<_FLOAT, 3> &tensor1,
                          const std::initializer_list<Eigen::Index> &start_idx);

template void
SetTensorSliceByTensor<4>(Eigen::Tensor<_FLOAT, 4> &tensor0,
                          const Eigen::Tensor<_FLOAT, 4> &tensor1,
                          const std::initializer_list<Eigen::Index> &start_idx);

template <typename T> void printTensor2D(std::string str, const T &tensor)
{
    using IndexType = typename T::Index;
    Eigen::array<IndexType, T::NumIndices> dims = tensor.dimensions();

    printf("%s", str.c_str());
    for (IndexType i = 0; i < dims[0]; ++i)
    {
        for (IndexType j = 0; j < dims[1]; ++j)
        {
            printf("value (%ld, %ld) = %.5f\n", i, j, tensor(i, j));
        }
    }
}

template <typename T> void printTensor3D(std::string str, const T &tensor)
{
    using IndexType = typename T::Index;
    Eigen::array<IndexType, T::NumIndices> dims = tensor.dimensions();

    printf("%s", str.c_str());
    for (IndexType i = 0; i < dims[0]; ++i)
    {
        for (IndexType j = 0; j < dims[1]; ++j)
        {
            for (IndexType k = 0; k < dims[2]; ++k)
            {
                printf("value (%ld, %ld, %ld) = %.5f\n", i, j, k, tensor(i, j, k));
            }
        }
    }
}

template <typename T> void printTensor4D(std::string str, const T &tensor)
{
    using IndexType = typename T::Index;
    Eigen::array<IndexType, T::NumIndices> dims = tensor.dimensions();

    printf("%s", str.c_str());
    for (IndexType i = 0; i < dims[0]; ++i)
    {
        for (IndexType j = 0; j < dims[1]; ++j)
        {
            for (IndexType k = 0; k < dims[2]; ++k)
            {
                for (IndexType l = 0; l < dims[3]; ++l)
                {
                    printf("value (%ld, %ld, %ld, %ld) = %.5f\n", i, j, k, l,
                           tensor(i, j, k, l));
                }
            }
        }
    }
}

template <typename T> void printTensor(std::string str, const T &tensor)
{
    if constexpr (T::NumIndices == 2)
    {
        printTensor2D<T>(str, tensor);
    }
    else if constexpr (T::NumIndices == 3)
    {
        printTensor3D<T>(str, tensor);
    }
    else if constexpr (T::NumIndices == 4)
    {
        printTensor4D<T>(str, tensor);
    }
    else
    {
        printf("cannot print %ld dim tensor\n", T::NumIndices);
    }
}

template void printTensor<tTensor2d>(std::string str, const tTensor2d &tensor);
template void printTensor<tTensor3d>(std::string str, const tTensor3d &tensor);
template void printTensor<tTensor4d>(std::string str, const tTensor4d &tensor);

} // namespace nTensorUtil
