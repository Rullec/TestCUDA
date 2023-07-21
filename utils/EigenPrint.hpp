#ifndef EIGEN_PRINT_HPP_
#define EIGEN_PRINT_HPP_
#include "utils/EigenUtil.h"
#include <iostream>
#include <string>

template <typename dtype, int M, int N>
void EIGEN_PRINT(const std::string &name, const Eigen::Matrix<dtype, M, N> &mat)
{
    if (N == 1)
    {
        std::cout << name << " = " << mat.transpose() << std::endl;
    }
    else
    {
        std::cout << name << " = \n" << mat << std::endl;
    }
}
#endif