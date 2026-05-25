#pragma once
#include "Eigen/Dense"
#include "autodiff/reverse/var/var.hpp"

namespace MiniBrain
{
    typedef Eigen::MatrixXf Matrix;
    typedef Eigen::VectorXf Vector;
    typedef Eigen::RowVectorXi IntVector;

    typedef Eigen::ArrayXf Array;
    //内存对齐映射，业务逻辑跟直接访问vector一样，只是效率更高
    typedef Eigen::VectorXf::ConstAlignedMapType ConstAlignedMapVec;
    typedef Eigen::VectorXf::AlignedMapType AlignedMapVec;

    typedef Eigen::MatrixXf::ConstAlignedMapType ConstAlignedMapMat;

    typedef float Scalar;

    typedef Eigen::Array<Scalar,1,Eigen::Dynamic> RowArray;

    typedef autodiff::Variable<Scalar> AutoDiffVar;

} // namespace MiniBrain
