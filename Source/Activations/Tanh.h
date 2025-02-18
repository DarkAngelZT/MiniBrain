#pragma once

#include "../Eigen/Dense"
#include "../Activation.h"

namespace MiniBrain
{
    class Tanh : public Activation
    {
    protected:
        /* data */
    public:
        Tanh(/* args */){}
        ~Tanh(){}

        virtual void Forward(const Matrix& InData) override
        {
            m_out.array() = InData.array().tanh();
        }

        // tanh'(x) = 1 - tanh(x)^2
        // J = d_a / d_z = diag(1 - a^2)
        // out = J * f = (1 - a^2) .* f
        virtual void Backward(const Matrix& InData,const Matrix& NextLayerData) override
        {
            m_din.array() = (float(1) - m_out.array().square())*NextLayerData.array();
        }

        virtual std::string GetSubType()const override{return "Tanh";}
    };
    
}