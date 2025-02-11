#pragma once
#include "Node.h"

namespace MiniBrain{
    class Activation: public Node
    {
    protected:
        /* data */
        Matrix m_out;
        Matrix m_din;
    public:
        Activation(/* args */) {}
        ~Activation() {}

        virtual void Forward(const Matrix& InData) = 0;

        virtual const Matrix& Output() {return m_out;}

        virtual void Backward(const Matrix& InData,const Matrix& NextLayerData) = 0;

        virtual const Matrix& GetBackpropData() {return m_din;}

        virtual std::string GetType()const override {return "Activation";}
    };
}