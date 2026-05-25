#ifndef ATTENTION_H
#define ATTENTION_H

#include "../Layer.h"
#include "../Optimizer.h"
#include <memory>
#include <vector>

namespace MiniBrain
{
    class Attention : public Layer
    {
        int d_feat;
        // 权重矩阵
        Matrix Wq, Wk, Wv;
        Matrix dWq, dWk, dWv;
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

        virtual void Forward(const Matrix& InData) override
        {
            
        }

        virtual void Update(Optimizer& opt) override
        {
            ConstAlignedMapVec dw(dWq.data(),dWq.size());
            ConstAlignedMapVec db(dWk.data(),dWk.size());
            ConstAlignedMapVec dv(dWv.data(),dWv.size());
            AlignedMapVec w(Wq.data(),Wq.size());
            AlignedMapVec b(Wk.data(),Wk.size());
            AlignedMapVec v(Wv.data(),Wv.size());
            opt.Update(dw,w);
            opt.Update(db,b);
            opt.Update(dv,v);
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