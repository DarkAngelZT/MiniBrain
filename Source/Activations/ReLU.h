#pragma once

#include "../Eigen/Dense"
#include "../Activation.h"

namespace MiniBrain
{
    template<typename T>
    class ReLU : public Activation<T>
    {
    protected:
        /* data */        
    public:
        ReLU(/* args */){}
        ~ReLU(){}

        virtual Matrix<T> Forward(const Matrix<T>& InData) override
        {
            
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                return InData.unaryExpr([](const AutoDiffVar& x){ return x.expr->val > Scalar(0) ? x : T(0); });
            }
            else
            {
                Matrix<T> m_out(InData.rows(), InData.cols());
                m_out.array() = InData.array().cwiseMax(Scalar(0));
                return m_out;
            }
            
        }

        // J = d_a / d_z = diag(sign(a)) = diag(a > 0)
        // out = J * f = (a > 0) .* f
        virtual void Backward(T& Loss) override
        {
            // m_din.array() = (m_out.array()>Scalar(0)).select(NextLayerData,Scalar(0));
        }

        virtual std::string GetSubType()const override{return "ReLU";}
    };
}