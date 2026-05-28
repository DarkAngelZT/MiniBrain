#pragma once
#include "TypeDef.h"
#include "Node.h"
#include "Eigen/Dense"

namespace MiniBrain
{
    class LossFunc : public Node
    {
    public:
        LossFunc(/* args */) {}
        ~LossFunc() {}

        virtual AutoDiffVar Evaluate(const Matrix<AutoDiffVar>& preLayerData, const Matrix<AutoDiffVar>& target) = 0;

        virtual std::string GetType() const override {return "LossFunc";}
    };
}