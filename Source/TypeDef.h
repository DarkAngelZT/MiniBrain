#pragma once
#include <cassert>
#include <exception>
#include "Eigen/Dense"
#include "autodiff/reverse/var/var.hpp"

#if defined(_CPPUNWIND) || defined(__cpp_exceptions) || defined(__EXCEPTIONS)
#  define MINIBRAIN_THROW(EX) throw EX
#else
#  define MINIBRAIN_THROW(EX) do { assert(false && "Exception disabled: " #EX); std::terminate(); } while(false)
#endif

namespace MiniBrain
{
    template<typename T>
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    template<typename T>
    using Vector = Eigen::VectorX<T>;

    typedef Eigen::RowVectorXi IntVector;

    template<typename T>
    using Array = Eigen::Array<T, Eigen::Dynamic, 1>;

    //内存对齐映射，业务逻辑跟直接访问vector一样，只是效率更高
    template<typename T>
    using ConstAlignedMapVec = Eigen::Map<const Eigen::VectorX<T>, Eigen::Aligned>;
    
    template<typename T>
    using AlignedMapVec = Eigen::Map<Eigen::VectorX<T>, Eigen::Aligned>;

    template<typename T>
    using ConstAlignedMapMat = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Aligned>;

    typedef float Scalar;

    typedef Eigen::Array<Scalar,1,Eigen::Dynamic> RowArray;

    typedef autodiff::Variable<Scalar> AutoDiffVar;

} // namespace MiniBrain
