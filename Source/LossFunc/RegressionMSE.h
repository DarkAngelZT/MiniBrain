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

        virtual AutoDiffVar Evaluate(const Matrix<AutoDiffVar>& Result, const Matrix<AutoDiffVar>& Target) override
        {
            const int nobs = Result.cols();
            const int nvar = Result.rows();
            if (Target.cols() != nobs || Target.rows() != nvar)
            {
                MINIBRAIN_THROW(std::invalid_argument("RegressionMSE: target data dimension mismatch"));
            }
            Matrix<AutoDiffVar> m_din(nvar, nobs);
            m_din = Result - Target;
            return m_din.squaredNorm() / (2 * m_din.cols());
        }

        virtual std::string GetSubType()const override{return "RegressionMSE";}
    };
} // namespace MiniBrain
