#pragma once
#include "Node.h"
#include "ComputeNode.h"
#include "Random.h"
#include <vector>

namespace MiniBrain {
    template<typename T>
    class Optimizer;
    
    template<typename T>
    class Layer: public IComputeNode<T>
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

        virtual void Init(const Scalar& mu, const Scalar& sigma, Random& RNG) = 0;

        virtual void Update(Optimizer<Scalar>& opt) = 0;

        // Fast training path: a Network can collect every trainable variable,
        // run autodiff::gradient() once for the whole loss graph, then send the
        // resulting slices back to their owning layers. Parameter-free layers
        // inherit these empty defaults.
        virtual int GetAutoDiffParameterCount() const { return 0; }

        virtual void AppendAutoDiffParameters(
            Vector<AutoDiffVar>& /* destination */,
            int& /* offset */) const
        {}

        virtual void AssignGradients(
            const Vector<Scalar>& /* gradients */,
            int& /* offset */)
        {}

        virtual std::vector<Scalar> GetParameters() const = 0;

        virtual void SetParameters(const std::vector<Scalar>& param) {};

        virtual bool HasParameters() const { return true; }

        virtual std::string GetType()const override {return "Layer";}        
    };
    
}
