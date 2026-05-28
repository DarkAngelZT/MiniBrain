#pragma once
#include "Eigen/Dense"
#include "TypeDef.h"
#include "Node.h"

namespace MiniBrain
{
    template<typename T>
    class IComputeNode : public Node
    {
        public:
            virtual Matrix<T> Forward(const Matrix<T>& InData) = 0;

            virtual void Backward(T& Loss) = 0;
    };
} // namespace MiniBrain
