#pragma once
#include "../Activation.h"
#include "../Eigen/Dense"

namespace MiniBrain
{
    template<typename T>
    class SoftMax : public Activation<T>
    {
    public:
        SoftMax(){}
        ~SoftMax(){}

        // a = activation(z) = softmax(z)
        virtual Matrix<T> Forward(const Matrix<T>& InData) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                // 1. 先把当前 InData 里的纯数字抽干（查表，不增加任何多余图节点）
                Eigen::MatrixX<Scalar> pure_data = InData.unaryExpr([](const AutoDiffVar& x) { return x.expr->val; });
                
                // 2. 完全对齐推理分支：算出每一列的最大值系数（维度：1 x Cols）
                Eigen::RowVectorX<Scalar> col_max = pure_data.colwise().maxCoeff();

                Matrix<T> exp_matrix(InData.rows(), InData.cols());

                for (int col = 0; col < InData.cols(); ++col) 
                {
                     exp_matrix.col(col) = (InData.col(col).array() - col_max(col)).exp();
                }

                
                return exp_matrix.array().rowwise() / exp_matrix.colwise().sum().array();
            }
            else
            {
                Matrix<T> m_out(InData.rows(), InData.cols());
                m_out.array() = (InData.rowwise() - InData.colwise().maxCoeff()).array().exp();
                RowArray colSum = m_out.colwise().sum();
                m_out.array().rowwise() /= colSum;
                return m_out;
            }
            
        }

        // J = d_a / d_z = diag(a) - a * a'
        // g = J * f = a .* f - a * (a' * f) = a .* (f - a'f)
        virtual void Backward(T& Loss) override
        {
            // RowArray aDotf = m_out.cwiseProduct(NextLayerBackpropData).colwise().sum();
            // m_din.array() = m_out.array()*(NextLayerBackpropData.array().rowwise()-aDotf);
        }

        virtual std::string GetSubType() const override { return "SoftMax"; }
    };
} // namespace MiniBrain
