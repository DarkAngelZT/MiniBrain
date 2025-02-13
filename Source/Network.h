#pragma once

#include "Layer.h"
#include "Activation.h"
#include "LossFunc.h"
#include "Optimizer.h"

namespace MiniBrain
{
    class Network
    {
    protected:
        std::vector<Layer *> m_layers;
        LossFunc *m_lossFunc;
    public:
        Network(/* args */) {}
        ~Network() {}

        void AddLayer(Layer *layer)
        {
            m_layers.push_back(layer);
        }

        void SetLossFunc(LossFunc *lossFunc)
        {
            m_lossFunc = lossFunc;
        }
    };
} // namespace MiniBrain
