#pragma once

#include "../Eigen/Dense"
#include "../Activation.h"

namespace MiniBrain
{
    class ReLU : public Activation
    {
    protected:
        /* data */        
    public:
        ReLU(/* args */){}
        ~ReLU(){}

        virtual void Forward(const Matrix& InData) override
        {
            m_out.array() = InData.array().cwiseMax(float(0));
        }

        // J = d_a / d_z = diag(sign(a)) = diag(a > 0)
        // out = J * f = (a > 0) .* f
        virtual void Backward(const Matrix& InData, const Matrix& NextLayerData) override
        {
            m_din.array() = (m_out.array()>float(0)).select(NextLayerData,float(0));
        }

        virtual std::string GetSubType()const override{return "ReLU";}
    };
}