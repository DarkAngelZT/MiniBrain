#pragma once
#include "Node.h"
#include "Eigen/Dense"

namespace MiniBrain
{
    typedef Eigen::MatrixXf Matrix;
    typedef Eigen::VectorXf Vector;
    typedef Eigen::RowVectorXi IntVector;

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

        virtual std::string GetType() {return "LossFunc";}
    };
}