#pragma once
#include "Eigen/Dense"

namespace MiniBrain {
    typedef Eigen::MatrixXf Matrix;
    typedef Eigen::VectorXf Vector;
    
    class Node {
        public:
            Node(){}

            virtual ~Node(){}

            virtual void Init() = 0;

            virtual void Forward(const Matrix& InData) = 0;

            virtual const Matrix& Output() const = 0;

            virtual void Backward(const Matrix& LastLayerData,const Matrix& NextLayerData) = 0;

            virtual const Matrix& GetBackpropData() const = 0;

            virtual void Update() = 0;
    };
}