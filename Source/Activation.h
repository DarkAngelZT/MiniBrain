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
        Matrix<T> m_out;
        Matrix<T> m_din;
    public:
        Activation(/* args */) {}
        ~Activation() {}

        virtual const Matrix<T>& Output() const override
        {
            return m_out;
        }

        virtual const Matrix<T>& GetBackpropData() const override
        {
            return m_din;
        }

        virtual std::string GetType()const override {return "Activation";}
    };
}