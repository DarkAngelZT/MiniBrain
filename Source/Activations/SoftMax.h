#pragma once
#include "../Activation.h"
#include "../Eigen/Dense"

namespace MiniBrain
{
    class SoftMax : public Activation
    {
    public:
        SoftMax(){}
        ~SoftMax(){}

        // a = activation(z) = softmax(z)
        virtual void Forward(const Matrix& InData) override
        {
            m_out.array() = (InData.rowwise() - InData.colwise().maxCoeff()).array().exp();
            RowArray colSum = m_out.colwise().sum();
            m_out.colwise() /= colSum;
        }

        // J = d_a / d_z = diag(a) - a * a'
        // g = J * f = a .* f - a * (a' * f) = a .* (f - a'f)
        virtual void Backward(const Matrix& Indata, const Matrix& NextLayerBackpropData) override
        {
            RowArray aDotf = m_out.cwiseProduct(NextLayerBackpropData).colwise().sum();
            m_din.array() = m_out.array()*(NextLayerBackpropData.array().rowwise()-aDotf);
        }

        virtual std::string GetSubType() const override { return "SoftMax"; }
    };
} // namespace MiniBrain
