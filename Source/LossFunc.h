#pragma once
#include "TypeDef.h"
#include "Node.h"
#include "Eigen/Dense"

namespace MiniBrain
{
    template<typename T>
    class LossFunc : public Node
    {
    protected:
        Matrix<T> m_din;
    public:
        LossFunc(/* args */) {}
        ~LossFunc() {}

        virtual void Evaluate(const Matrix<T>& preLayerData, const Matrix<T>& target) = 0;

        virtual const Matrix<T>& GetBackpropData() const {return m_din;}

        virtual Scalar GetLoss() const = 0;

        virtual std::string GetType() const override {return "LossFunc";}
    };
}