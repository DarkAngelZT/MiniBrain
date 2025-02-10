#pragma once
#include "Eigen/Dense"

namespace MiniBrain {
    typedef Eigen::MatrixXf Matrix;
    typedef Eigen::VectorXf Vector;
    
    class Node {
        public:
            Node(){}

            virtual ~Node(){}
            
            virtual std::string GetType()const {return "Node";}
    };
}