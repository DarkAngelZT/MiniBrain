#pragma once
#include "Node.h"
#include "ComputeNode.h"
#include "Random.h"
#include <vector>

namespace MiniBrain {
    class Optimizer;
    
    class Layer: public IComputeNode
    {
    private:
        /* data */
    protected:
        int m_inSize,m_outSize;
    public:
        Layer(int inSize,int OutSize):m_inSize(inSize),m_outSize(OutSize)
        {}
        virtual~Layer() {}

        int GetInSize() const {return m_inSize;}
        int GetOutSize() const {return m_outSize;}

        virtual void Init() = 0;

        virtual void Init(const float& mu, const float& sigma, Random& RNG) = 0;

        virtual void Update(Optimizer& opt) = 0;

        virtual std::vector<float> GetParameters() const = 0;

        virtual void SetParameters(const std::vector<float>& param) {};

        virtual std::string GetType()const override {return "Layer";}        
    };
    
}