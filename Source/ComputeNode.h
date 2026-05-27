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

            virtual void Backward(const Matrix<T>& LastLayerData,const Matrix<T>& NextLayerData) = 0;

            virtual const Matrix<T>& GetBackpropData() const = 0;
    };
} // namespace MiniBrain
