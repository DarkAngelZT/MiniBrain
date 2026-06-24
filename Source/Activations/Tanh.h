#pragma once

#include "../Eigen/Dense"
#include "../Activation.h"

namespace MiniBrain
{
    template<typename T>
    class Tanh : public Activation<T>
    {
    protected:
        /* data */
    public:
        Tanh(/* args */){}
        ~Tanh(){}

        virtual Matrix<T> Forward(const Matrix<T>& InData) override
        {
            using autodiff::reverse::detail::tanh;
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                return InData.unaryExpr([](const AutoDiffVar& x){ return tanh(x); });
            }
            else
            {
                Matrix<T> m_out(InData.rows(), InData.cols());
                m_out.array() = InData.array().tanh();
                return m_out;
            }
        }

        // tanh'(x) = 1 - tanh(x)^2
        // J = d_a / d_z = diag(1 - a^2)
        // out = J * f = (1 - a^2) .* f
        virtual void Backward(T& Loss) override
        {
            // m_din.array() = (Scalar(1) - m_out.array().square())*NextLayerData.array();
        }

        virtual std::string GetSubType()const override{return "Tanh";}
    };
    
}