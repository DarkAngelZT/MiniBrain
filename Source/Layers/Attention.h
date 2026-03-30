#ifndef ATTENTION_H
#define ATTENTION_H

#include "../Layer.h"
#include <memory>
#include <vector>

namespace MiniBrain
{
    class Attention : public Layer
    {

    public:
        Attention(int inputSize, int outputSize, int headCount):Layer(inputSize, outputSize)
        {
            Init();
        }
        
        virtual void Init() override
        {

        }

        virtual void Init(const float& mu, const float& sigma, Random& RNG) override
        {

        }

        virtual void Forward(const Matrix& InData) override
        {
            
        }

        virtual void Update(Optimizer& opt) override
        {

        }

        virtual std::vector<float> GetParameters() const override
        {
            return std::vector<float>();
        }

        virtual void SetParameters(const std::vector<float>& param) override 
        {
            
        };

        virtual std::string GetSubType()const override{return "Attention";} 
    };
}

#endif // ATTENTION_H