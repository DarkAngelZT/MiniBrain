#pragma once
#include "TypeDef.h"
#include "Layer.h"
#include "Activation.h"
#include "LossFunc.h"
#include "Optimizer.h"
#include <vector>

namespace MiniBrain
{
    class Network
    {
    protected:
        std::vector<IComputeNode *> m_layers;
        LossFunc *m_lossFunc=nullptr;
        Random m_defaultRNG;
        Random& m_rng;

    public:
        bool CheckUnitSize() const
        {
            const int numLayer = GetLayerAmount();
            if (numLayer <= 1)
            {
                return true;
            }
            int i = 0;
            while (i < numLayer && m_layers[i]->GetType()!="Layer")
            {
                i++;
            }
            for (i = i + 1; i < numLayer; i++)
            {
                int j = i - 1;
                while (i < numLayer && m_layers[i]->GetType()!="Layer")
                {
                    i++;
                }
                if (i>=numLayer)
                {
                    return true;
                }
                if (static_cast<Layer*>(m_layers[i])->GetInSize() != static_cast<Layer*>(m_layers[j])->GetOutSize())
                {
                    return false;
                }                                
            }
            
            return true;
        }

        virtual void Forward(const Matrix& InData)
        {
            const int numLayers = GetLayerAmount();
            if (numLayers <= 0)
            {
                return;
            }
            
            if(InData.rows() != static_cast<Layer*>(m_layers[0])->GetInSize())
            {
                throw std::invalid_argument("[Network]: Input data dimension mismatch");
            }

            m_layers[0]->Forward(InData);

            for (int i = 1; i < numLayers; i++)
            {
                m_layers[i]->Forward(m_layers[i-1]->Output());
            }
            
        }

        virtual void Backward(const Matrix& Input, const Matrix& Target)
        {
            const int nLayer = GetLayerAmount();
            if (nLayer <= 0)
            {
                return;
            }
            
            IComputeNode* FirstLayer = m_layers[0];
            IComputeNode* LastLayer = m_layers[nLayer-1];

            m_lossFunc->Evaluate(LastLayer->Output(),Target);
            
            if (nLayer == 1)
            {
                FirstLayer->Backward(Input,m_lossFunc->GetBackpropData());
                return;
            }
            
            LastLayer->Backward(m_layers[nLayer-2]->Output(), m_lossFunc->GetBackpropData());

            for (int i = nLayer-2; i >0; i--)
            {
                m_layers[i]->Backward(m_layers[i - 1]->Output(), m_layers[i + 1]->GetBackpropData());
            }
            
            FirstLayer->Backward(Input, m_layers[1]->GetBackpropData());
        }

        virtual void Update(Optimizer& opt)
        {
            const int nLayer = GetLayerAmount();
            if (nLayer <= 0)
            {
                return;
            }

            for (int i = 0; i < nLayer; i++)
            {
                if (m_layers[i]->GetType() == "Layer")
                {
                    dynamic_cast<Layer*>(m_layers[i])->Update(opt);
                }                
            }            
        }

        Network() :
            m_lossFunc(nullptr),
            m_rng(m_defaultRNG)
        {}
        ~Network() 
        {
            for (int i = 0; i < GetLayerAmount(); i++)
            {
                delete m_layers[i];
            }
            if(m_lossFunc)
            {
                delete m_lossFunc;
            }
        }

        virtual void Init(const float& mu=0.f, const float& sigma=1.f)
        {
            if(!CheckUnitSize())
            {
                throw std::invalid_argument("[Network]Layer size mismatch");
            }
            const int nLayers = GetLayerAmount();
            
            for (int i = 0; i < nLayers; i++)
            {
                if (m_layers[i]->GetType()=="Layer")
                {
                    dynamic_cast<Layer*>(m_layers[i])->Init(mu,sigma,m_rng);
                }                
            }
            
        }

        void AddLayer(IComputeNode *layer)
        {
            m_layers.push_back(layer);
        }

        void SetLossFunc(LossFunc *lossFunc)
        {
            m_lossFunc = lossFunc;
        }

        int GetLayerAmount() const
        {
            return m_layers.size();
        }

        const LossFunc* GetLossFunc() const
        {
            return m_lossFunc;
        }

        virtual Matrix Predict(const Matrix& Input)
        {
            const int nLayer = GetLayerAmount();

            if (nLayer <= 0)
            {
                return Matrix();
            }
            
            Forward(Input);

            return m_layers[nLayer-1]->Output();
        }
        
    };
} // namespace MiniBrain
