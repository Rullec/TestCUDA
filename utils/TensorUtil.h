#ifndef TENSOR_UTIL_H
#define TENSOR_UTIL_H
#include "utils/BaseTypeUtil.h"
#include "utils/EigenUtil.h"
#include <unsupported/Eigen/CXX11/Tensor>
typedef Eigen::Tensor<_FLOAT, 1> tTensor1d;
typedef Eigen::Tensor<_FLOAT, 2> tTensor2d;
typedef Eigen::Tensor<_FLOAT, 3> tTensor3d;
typedef Eigen::Tensor<_FLOAT, 4> tTensor4d;

namespace nTensorUtil
{

template <int N>
Eigen::array<long, N> CreateTensorIndex(const std::initializer_list<long> &arr);

template <typename dtype, int N>
tVectorXi GetTensorDim(const Eigen::Tensor<dtype, N> &tensor);

Eigen::Tensor<_FLOAT, 1> ConvertVectorToTensor(const tVectorX &mat);
Eigen::Tensor<_FLOAT, 2> ConvertMatrixToTensor(const tMatrixX &mat);
tMatrixX ConvertTensorToMatrix(const tTensor2d &t);
tVectorX ConvertTensorToVector(const tTensor1d &t);

template <typename T, int NumDims, int NumNewDims>
Eigen::Tensor<T, NumDims + NumNewDims>
TensorExpand(const Eigen::Tensor<T, NumDims> &input_tensor,
             const std::vector<int> &positions);

template <int N>
void SetTensorSliceByTensor(
    Eigen::Tensor<_FLOAT, N> &tensor0, const Eigen::Tensor<_FLOAT, N> &tensor1,
    const std::initializer_list<Eigen::Index> &start_idx);

template <typename T, int Tensor1Rank, int Tensor2Rank, int NumContractions>
Eigen::Tensor<T, Tensor1Rank + Tensor2Rank - 2 * NumContractions>
TensorContract(const Eigen::Tensor<T, Tensor1Rank> &tensor1,
               const Eigen::Tensor<T, Tensor2Rank> &tensor2,
               std::initializer_list<Eigen::IndexPair<int>> contraction_pairs)
{
    Eigen::array<Eigen::IndexPair<int>, NumContractions> contraction_dims;
    std::copy(contraction_pairs.begin(), contraction_pairs.end(),
              contraction_dims.begin());
    return tensor1.contract(tensor2, contraction_dims);
}

template <typename T, int Tensor1Rank, int Tensor2Rank, int Tensor3Rank>
Eigen::Tensor<T, Tensor1Rank + Tensor2Rank + Tensor3Rank - 4>
TensorContractABC(const Eigen::Tensor<T, Tensor1Rank> &tensor1,
                  const Eigen::Tensor<T, Tensor2Rank> &tensor2,
                  const Eigen::Tensor<T, Tensor3Rank> &tensor3,
                  const std::vector<int> &pair0, const std::vector<int> &pair1)
{
    return TensorContract<T, Tensor1Rank + Tensor2Rank - 2, Tensor3Rank, 1>(
        TensorContract<T, Tensor1Rank, Tensor2Rank, 1>(tensor1, tensor2,
                                                       {{pair0[0], pair0[1]}}),
        tensor3, {{pair1[0], pair1[1]}});
}

template <typename T> void printTensor(std::string str, const T &tensor);
} // namespace nTensorUtil
#endif