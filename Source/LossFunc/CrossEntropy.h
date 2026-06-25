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
        virtual AutoDiffVar Evaluate(const Matrix<AutoDiffVar>& Result, const Matrix<AutoDiffVar>& Target) override
        {
            const int nobs = Result.cols();
            const int nvar = Result.rows();
            Scalar batch_size = static_cast<float>(nobs);
            if (Target.cols() != nobs || Target.rows() != nvar)
            {
                MINIBRAIN_THROW(std::invalid_argument("CrossEntropy_Multi: target data dimension mismatch"));
            }
            // Compute the derivative of the input of this layer
            // L = -sum(log(phat) * y)
            // in = phat
            // d(L) / d(in) = -y / phat
            //以上手工推导逻辑仅做参考，现在换自动微分了，可以直接照搬公式

            AutoDiffVar total_loss = 0.0f;
            const Scalar eps = 1e-7f; // 防止 log(0) 导致的数值不稳定

            Matrix<AutoDiffVar> clipped_result = Result.unaryExpr([eps](const AutoDiffVar& x) {
                return x.expr->val < eps ? AutoDiffVar(eps) : x;
            });

            // 直接利用 Eigen 的一元数组函数和点乘，一行代码算完整个 Batch 的交叉熵
            // .array().log() 会自动触发 autodiff 的全局 log 重载，完美录制计算图
            // 最后的 .sum() 瞬间把整个二维矩阵的误差聚合成一个孤立的 AutoDiffVar 标量根节点
            total_loss = -(Target.array() * clipped_result.array().log()).sum();
            return total_loss / batch_size;
        }

        virtual std::string GetSubType()const override{return "CrossEntropy_Multi";}
    };
} // namespace MiniBrain
