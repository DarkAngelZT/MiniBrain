#pragma once
#include "Node.h"

namespace MiniBrain {
    class Layer: public Node
    {
    private:
        /* data */
    protected:
        int m_inSize,m_outSize;
    public:
        Layer(int inSize,int OutSize):m_inSize(inSize),m_outSize(OutSize)
        {}
        virtual~Layer() {}

    };
    
}