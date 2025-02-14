#pragma once
#include "ComputeNode.h"
#include "TypeDef.h"
#include "Node.h"

namespace MiniBrain{
    class Activation: public IComputeNode
    {
    protected:
        /* data */
        Matrix m_out;
        Matrix m_din;
    public:
        Activation(/* args */) {}
        ~Activation() {}

        virtual const Matrix& Output() const override
        {
            return m_out;
        }

        virtual const Matrix& GetBackpropData() const override
        {
            return m_din;
        }

        virtual std::string GetType()const override {return "Activation";}
    };
}