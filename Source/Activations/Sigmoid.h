#pragma once

#include "../Activation.h"
#include "../Eigen/Dense"

namespace MiniBrain
{
    class Sigmoid : public Activation
    {
    protected:
        /* data */
    public:
        Sigmoid(/* args */) {}
        ~Sigmoid() {}

        //activation(z) = 1 / (1 + exp(-z))
        virtual void Forward(const Matrix& InData) override
        {
            m_out.array() = 1.0f/(1.0f + (-InData.array()).exp());
        }

        // J = d_a / d_z = diag(a .* (1 - a))
        // g = J * f = a .* (1 - a) .* f
        virtual void Backward(const Matrix& Indata, const Matrix& NextLayerBackpropData) override
        {
            m_din.array() = m_out.array() * (1.0f - m_out.array())*NextLayerBackpropData.array();
        }

        virtual std::string GetSubType() const override { return "Sigmoid"; }
    };
} // namespace MiniBrain
