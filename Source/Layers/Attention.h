#ifndef ATTENTION_H
#define ATTENTION_H
#include <autodiff/reverse/var/eigen.hpp>
#include "../Layer.h"
#include "../Optimizer.h"
#include <memory>
#include <vector>

namespace MiniBrain
{
    template<typename T>
    class Attention : public Layer<T>
    {
        int d_feat;
        // 权重矩阵
        Matrix<T> Wq, Wk, Wv;
        Matrix<Scalar> dWq, dWk, dWv;
    public:
        Attention(int inputSize, int outputSize, int featureCount):Layer(inputSize, outputSize)
        {
            d_feat = inputSize / featureCount;
            Init();
        }
        
        virtual void Init() override
        {

        }

        virtual void Init(const Scalar& mu, const Scalar& sigma, Random& RNG) override
        {

        }

        virtual Matrix<T> Forward(const Matrix<T>& InData) override
        {
            
        }

        void Backward(T& Loss) override
        {
            
        }

        virtual void Update(Optimizer<Scalar>& opt) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                opt.Update(dWq, Wq);
                opt.Update(dWk, Wk);
                opt.Update(dWv, Wv);
            }
        }

        virtual std::vector<Scalar> GetParameters() const override
        {
            return std::vector<Scalar>();
        }

        virtual void SetParameters(const std::vector<Scalar>& param) override 
        {
            
        };

        virtual std::string GetSubType()const override{return "Attention";} 
    };
}

#endif // ATTENTION_H