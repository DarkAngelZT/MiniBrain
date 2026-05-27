#pragma once
#include "ComputeNode.h"
#include "TypeDef.h"
#include "Node.h"

namespace MiniBrain{
    template<typename T>
    class Activation: public IComputeNode<T>
    {
    protected:
        /* data */
        // Matrix<T> m_out;
        // Matrix<T> m_din;
    public:
        Activation(/* args */) {}
        ~Activation() {}

        virtual std::string GetType()const override {return "Activation";}
    };
}