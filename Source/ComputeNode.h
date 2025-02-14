#pragma once
#include "Eigen/Dense"
#include "TypeDef.h"
#include "Node.h"

namespace MiniBrain
{
    class IComputeNode : public Node
    {
        public:
            virtual void Forward(const Matrix& InData) = 0;
            
            virtual const Matrix& Output() const = 0;

            virtual void Backward(const Matrix& LastLayerData,const Matrix& NextLayerData) = 0;

            virtual const Matrix& GetBackpropData() const = 0;
    };
} // namespace MiniBrain
