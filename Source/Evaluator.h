#pragma once
#include "Eigen/Dense"

namespace MiniBrain
{
    typedef Eigen::MatrixXf Matrix;
    typedef Eigen::VectorXf Vector;
    typedef Eigen::RowVectorXi IntVector;

    class Evaluator
    {
    protected:
        /* data */
    public:
        Evaluator(/* args */) {}
        ~Evaluator() {}

        virtual void Evaluate(const Matrix&preLayerData, const Matrix& target) = 0;

        virtual const Matrix& GetBackpropData() const = 0;

        virtual float GetLoss() const = 0;

        virtual std::string GetType() {return "Evaluator";}
    };
}