#pragma once
#include "Eigen/Dense"
#include "TypeDef.h"

namespace MiniBrain
{
    template<typename T>
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
        virtual void Update(ConstAlignedMapVec<T>& dvec, AlignedMapVec<T>& vec) = 0;

        void Update(const Matrix<T>& Indw, Matrix<T>& Weight)
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                Matrix<Scalar> dW(Indw.rows(), Indw.cols());
                Matrix<Scalar> w(Weight.rows(), Weight.cols());
                dW = Indw.unaryExpr([](const AutoDiffVar& x){ return x.expr->val; });
                w = Weight.unaryExpr([](const AutoDiffVar& x){ return x.expr->val; });
                ConstAlignedMapVec<Scalar> dvec(dW.data(), dW.size());
                AlignedMapVec<Scalar> vec(w.data(), w.size());
                Update(dvec, vec);
                // Update the original weight matrix with the new values
                Weight = w.cast<AutoDiffVar>();
            }
        }
    };
    
}
