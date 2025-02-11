#pragma once

#include "../Eigen/Dense"
#include "../Activation.h"

namespace MiniBrain
{
    class Mish : public Activation
    {
    protected:
        /* data */
    public:
        Mish(/* args */){}
        ~Mish(){}

        // Mish(x) = x * tanh(softplus(x))
        // softplus(x) = log(1 + exp(x))
        virtual void Forward(const Matrix& InData) override
        {
            // h(x) = tanh(softplus(x)) = (1 + exp(x))^2 - 1
            //                            ------------------
            //                            (1 + exp(x))^2 + 1
            // Let s = exp(-abs(x)), t = 1 + s
            // If x >= 0, then h(x) = (t^2 - s^2) / (t^2 + s^2)
            // If x <= 0, then h(x) = (t^2 - 1) / (t^2 + 1)
            Matrix S = (-InData.array().abs()).exp();

            //t^2
            m_out.array() = (S.array()+float(1)).square();
            //s^2 or 1
            S.noalias() = (InData.array() >= float(0)).select(S.cwiseAbs2(),float(1));
            m_out.array() = (m_out.array()-S.array())/(m_out.array()+S.array());
            m_out.array() *= InData.array();
        }

        // J = d_a / d_z = diag(Mish'(z))
        // out = J * f = Mish'(z) .* f
        virtual void Backward(const Matrix& InData, const Matrix& NextLayerData) override
        {
            // Let h(x) = tanh(softplus(x))
            // Mish'(x) = h(x) + x * h'(x)
            // h'(x) = tanh'(softplus(x)) * softplus'(x)
            //       = [1 - h(x)^2] * exp(x) / (1 + exp(x))
            //       = [1 - h(x)^2] / (1 + exp(-x))
            // Mish'(x) = h(x) + [x - Mish(x) * h(x)] / (1 + exp(-x))
            
            // m_out = Mish(InData) = InData .* h(InData) => h(InData) = m_out ./ InData, h(0) = 0.6
            //get h(x)
            m_din.noalias() = (InData.array() == 0.f).select(0.6f,m_out.cwiseQuotient(InData));
            //get Mish'(x)
            m_din.array() = (InData.array()-m_out.array()*m_din.array())/(1.0f+(-InData).array().exp());
            //get da.*f
            m_din.array() *= NextLayerData.array();
        }

        virtual std::string GetSubType()const override{return "Mish";}

    };
    
}