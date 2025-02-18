#pragma once
#include "../Layer.h"

namespace MiniBrain
{
    class GRU : public Layer
    {
    protected:
        Matrix m_z;
        Matrix m_r;
        Matrix m_h;
        Matrix m_h_prev;
        Matrix m_weight_z;
        Matrix m_weight_r;
        Matrix m_weight_h;
        Matrix m_Uz;
        Matrix m_Ur;
        Matrix m_Uh;
        Matrix m_bias_z;
        Matrix m_bias_r;
        Matrix m_bias_h;
    public:
        GRU (int inSize,int OutSize):Layer(inSize,OutSize)
        {

        }
        ~GRU () {}

    };
} // namespace MiniBrain
