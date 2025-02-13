#pragma once
#include "../Eigen/Dense"
#include "../LossFunc.h"

namespace MiniBrain
{
    class CrossEntropy_Multi : public LossFunc
    {
    protected:
  
    public:
        CrossEntropy_Multi(/* args */) {}
        ~CrossEntropy_Multi() {}

        // target is a matrix with each column representing an observation
        virtual void Evaluate(const Matrix& Result, const Matrix& Target) override
        {
            const int nobs = Result.cols();
            const int nvar = Result.rows();
            if (Target.cols() != nobs || Target.rows() != nvar)
            {
                throw std::invalid_argument("RegressionMSE: target data dimension mismatch");
            }
            // Compute the derivative of the input of this layer
            // L = -sum(log(phat) * y)
            // in = phat
            // d(L) / d(in) = -y / phat
            m_din.resize(nvar,nobs);
            m_din.noalias() = -Target.cwiseQuotient(Result);
        }

        virtual float GetLoss() const override
        {
            float r = 0.f;
            const int nelem = m_din.size();
            const float* din_data = m_din.data();

            for (int i = 0; i < nelem; i++)
            {
                if(din_data[i] < 0.f)
                {
                    r += std::log(-din_data[i]);
                }
            }
            return r/m_din.cols();
        }

        virtual std::string GetSubType()const override{return "CrossEntropy_Multi";}
    };
} // namespace MiniBrain
