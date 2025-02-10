#pragma once
#include "Node.h"

namespace MiniBrain{
    class Activation: public Node
    {
    private:
        /* data */
    public:
        Activation(/* args */) {}
        ~Activation() {}

        virtual void Init() = 0;

        virtual void Forward(const Matrix& InData, Matrix& OutData) = 0;

        virtual void Backward(const Matrix& InData,const Matrix& LastOutputData,const Matrix& NextLayerData, Matrix& OutData) = 0;

        virtual std::string GetType()const {return "Activation";}

    };
}