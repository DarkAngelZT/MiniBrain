#pragma once

#include "../Eigen/Dense"
#include "../Activation.h"

namespace MiniBrain
{
    template<typename T>
    class Mish : public Activation<T>
    {
    protected:
        /* data */
    public:
        Mish(/* args */){}
        ~Mish(){}

        // Mish(x) = x * tanh(softplus(x))
        // softplus(x) = log(1 + exp(x))
        virtual void Forward(const Matrix<T>& InData) override
        {
            using autodiff::exp;
            using autodiff::log;
            using autodiff::tanh;
            // h(x) = tanh(softplus(x)) = (1 + exp(x))^2 - 1
            //                            ------------------
            //                            (1 + exp(x))^2 + 1
            // Let s = exp(-abs(x)), t = 1 + s
            // If x >= 0, then h(x) = (t^2 - s^2) / (t^2 + s^2)
            // If x <= 0, then h(x) = (t^2 - 1) / (t^2 + 1)
            constexpr float threshold = std::is_same_v<Scalar, float> ? 20.0f : 37.0f;
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                return InData.unaryExpr([](const AutoDiffVar& x){
                    if (x.expr->val > threshold) 
                    {
                        // 当 x 很大时，log(1 + exp(x)) 已经无限逼近于 x 本身
                        // 此时 tanh(x) 就是精确的激活值，直接省去 exp 运算，效率和安全度拉满！
                        return x * tanh(x);
                    } 
                    else if (x.expr->val < -threshold) 
                    {
                        // 当 x 很小时，log(1 + exp(x)) 无限逼近于 0，tanh(0) = 0，直接返回干净无图的 0
                        return AutoDiffVar(0.0f);
                    }
                    else 
                    {
                        AutoDiffVar h = tanh(log(1.0f + exp(x))); // 触发 autodiff 的全局 log 重载，录制计算图
                        return x * h;
                    }
                });
            }
            else
            {
                Matrix<T> S = (-InData.array().abs()).exp();
                Matrix<T> m_out(InData.rows(), InData.cols());
                //t^2
                m_out.array() = (S.array()+Scalar(1)).square();
                //s^2 or 1
                S.noalias() = (InData.array() >= Scalar(0)).select(S.cwiseAbs2(),Scalar(1));
                m_out.array() = (m_out.array()-S.array())/(m_out.array()+S.array());
                m_out.array() *= InData.array();
                return m_out;
            }
            
        }

        // J = d_a / d_z = diag(Mish'(z))
        // out = J * f = Mish'(z) .* f
        virtual void Backward(T& Loss) override
        {
            // Let h(x) = tanh(softplus(x))
            // Mish'(x) = h(x) + x * h'(x)
            // h'(x) = tanh'(softplus(x)) * softplus'(x)
            //       = [1 - h(x)^2] * exp(x) / (1 + exp(x))
            //       = [1 - h(x)^2] / (1 + exp(-x))
            // Mish'(x) = h(x) + [x - Mish(x) * h(x)] / (1 + exp(-x))
            
            // m_out = Mish(InData) = InData .* h(InData) => h(InData) = m_out ./ InData, h(0) = 0.6
            //get h(x)
            // m_din.noalias() = (InData.array() == 0.f).select(0.6f,m_out.cwiseQuotient(InData));
            //get Mish'(x)
            // m_din.array() = (InData.array()-m_out.array()*m_din.array())/(1.0f+(-InData).array().exp());
            //get da.*f
            // m_din.array() *= NextLayerData.array();
        }

        virtual std::string GetSubType()const override{return "Mish";}

    };
    
}