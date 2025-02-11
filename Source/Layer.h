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

        virtual void Init() = 0;

        virtual void Forward(const Matrix& InData) = 0;

        virtual const Matrix& Output() const = 0;

        virtual void Backward(const Matrix& LastLayerData,const Matrix& NextLayerData) = 0;

        virtual const Matrix& GetBackpropData() const = 0;

        virtual void Update() = 0;

        virtual std::vector<float> get_parameters() const = 0;

        virtual void set_parameters(const std::vector<float>& param) {};

        virtual std::string GetType()const override {return "Layer";}        
    };
    
}