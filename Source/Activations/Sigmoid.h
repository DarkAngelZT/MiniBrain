#pragma once

#include "../Activation.h"
#include "../Eigen/Dense"

namespace MiniBrain
{
    template<typename T>
    class Sigmoid : public Activation<T>
    {
    protected:
        /* data */
    public:
        Sigmoid(/* args */) {}
        ~Sigmoid() {}

        //activation(z) = 1 / (1 + exp(-z))
        virtual Matrix<T> Forward(const Matrix<T>& InData) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                return InData.unaryExpr([](const AutoDiffVar& x) -> AutoDiffVar { return 1.0f / (1.0f + autodiff::reverse::detail::exp(-x)); });
            }
            else
            {
                Matrix<T> m_out(InData.rows(), InData.cols());
                m_out.array() = 1.0f/(1.0f + (-InData.array()).exp());
                return m_out;
            }
        }

        // J = d_a / d_z = diag(a .* (1 - a))
        // g = J * f = a .* (1 - a) .* f
        virtual void Backward(T& Loss) override
        {
            // m_din.array() = m_out.array() * (1.0f - m_out.array())*NextLayerBackpropData.array();
        }

        virtual std::string GetSubType() const override { return "Sigmoid"; }
    };
} // namespace MiniBrain
