#pragma once
#include "../Eigen/Dense"
#include "../LossFunc.h"

namespace MiniBrain
{
    class RegressionMSE : public LossFunc
    {
    protected:
        /* data */
    public:
        RegressionMSE(/* args */) {}
        ~RegressionMSE() {}

        virtual void Evaluate(const Matrix& Result, const Matrix& Target) override
        {
            const int nobs = Result.cols();
            const int nvar = Result.rows();
            if (Target.cols() != nobs || Target.rows() != nvar)
            {
                throw std::invalid_argument("RegressionMSE: target data dimension mismatch");
            }
            m_din.resize(nvar,nobs);
            m_din.noalias() = Result - Target;
        }

        virtual float GetLoss() const override
        {
            //devide by 2, thus remove number 2 in derivative
            return m_din.squaredNorm()/m_din.cols() * 0.5f;
        }

        virtual std::string GetSubType()const override{return "RegressionMSE";}
    };
} // namespace MiniBrain
