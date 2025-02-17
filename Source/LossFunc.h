#pragma once
#include "TypeDef.h"
#include "Node.h"
#include "Eigen/Dense"

namespace MiniBrain
{
    class LossFunc : public Node
    {
    protected:
        Matrix m_din;
    public:
        LossFunc(/* args */) {}
        ~LossFunc() {}

        virtual void Evaluate(const Matrix& preLayerData, const Matrix& target) = 0;

        virtual const Matrix& GetBackpropData() const {return m_din;}

        virtual float GetLoss() const = 0;

        virtual std::string GetType() const override {return "LossFunc";}
    };
}