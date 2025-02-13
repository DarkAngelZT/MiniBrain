#pragma once
#include "Eigen/Dense"

namespace MiniBrain
{
    typedef Eigen::ArrayXf Array;
    //内存对齐映射，业务逻辑跟直接访问vector一样，只是效率更高
    typedef Eigen::VectorXf::ConstAlignedMapType ConstAlignedMapVec;
    typedef Eigen::VectorXf::AlignedMapType AlignedMapVec;

    class Optimizer
    {
    protected:
        /* data */
    public:
        Optimizer(/* args */){}
        ~Optimizer(){}

        virtual void Reset(){}

        ///
        /// Update the parameter vector using its gradient
        ///
        /// It is assumed that the memory addresses of `dvec` and `vec` do not
        /// change during the training process. This is used to implement optimization
        ///
        /// \param dvec The gradient of the parameter. Read-only
        /// \param vec  On entering, the current parameter vector. On exit, the
        ///             updated parameters.
        ///
        virtual void Update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) = 0;
    };
    
}
