#pragma once
#include "TypeDef.h"
#include "Layer.h"
#include "Activation.h"
#include "LossFunc.h"
#include "Optimizer.h"
#include <vector>

namespace MiniBrain
{
    template<typename T>
    class Network
    {
    protected:
        std::vector<IComputeNode<T> *> m_layers;
        LossFunc<T> *m_lossFunc=nullptr;
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

        virtual Matrix<T> Forward(const Matrix<T>& InData)
        {
            const int numLayers = GetLayerAmount();
            if (numLayers <= 0)
            {
                return;
            }
            
            if(InData.rows() != static_cast<Layer<T>*>(m_layers[0])->GetInSize())
            {
                throw std::invalid_argument("[Network]: Input data dimension mismatch");
            }

            Matrix<T> output = m_layers[0]->Forward(InData);

            for (int i = 1; i < numLayers; i++)
            {
                output = m_layers[i]->Forward(output);
            }
            return output;
        }

        virtual void Backward(const Matrix<T>& Input, const Matrix<T>& Target)
        {
            const int nLayer = GetLayerAmount();
            if (nLayer <= 0)
            {
                return;
            }
            
            IComputeNode<T>* FirstLayer = m_layers[0];
            IComputeNode<T>* LastLayer = m_layers[nLayer-1];

            m_lossFunc->Evaluate(LastLayer->Output(),Target);
            
            if (nLayer == 1)
            {
                FirstLayer->Backward(Input,m_lossFunc->GetBackpropData());
                return;
            }
            
            T loss = m_lossFunc->GetLoss();

            for (int i = nLayer-1; i >=0; i--)
            {
                m_layers[i]->Backward(loss);
            }
        }

        virtual void Update(Optimizer<T>& opt)
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
                    dynamic_cast<Layer<T>*>(m_layers[i])->Update(opt);
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

        virtual void Init(const Scalar& mu=0.f, const Scalar& sigma=1.f)
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
                    dynamic_cast<Layer<T>*>(m_layers[i])->Init(mu,sigma,m_rng);
                }                
            }
            
        }

        void AddLayer(IComputeNode<T> *layer)
        {
            m_layers.push_back(layer);
        }

        void SetLossFunc(LossFunc<T> *lossFunc)
        {
            m_lossFunc = lossFunc;
        }

        int GetLayerAmount() const
        {
            return m_layers.size();
        }

        const LossFunc<T>* GetLossFunc() const
        {
            return m_lossFunc;
        }

        virtual Matrix<T> Predict(const Matrix<T>& Input)
        {
            const int nLayer = GetLayerAmount();

            if (nLayer <= 0)
            {
                return Matrix<T>();
            }
            
            Matrix<T> output = Forward(Input);

            return output;
        }

        virtual std::vector<std::vector<Scalar>> GetParameters()const
        {
            const int nLayer = GetLayerAmount();
            std::vector<std::vector<Scalar>> result;
            result.reserve(nLayer);

            for (int i = 0; i < nLayer; i++)
            {
                if (m_layers[i]->GetType()=="Layer")
                {
                    result.push_back(dynamic_cast<Layer<T>*>(m_layers[i])->GetParameters());
                }               
            }            

            return result;
        }

        virtual void SetParameters(const std::vector<std::vector<Scalar>>& params)
        {
            const int nLayer = GetLayerAmount();
            if (static_cast<int>(params.size())!=nLayer)
            {
                std::invalid_argument("[network]:parameter size mismatch");
            }
            
            for (int i = 0; i < nLayer; i++)
            {
                if (m_layers[i]->GetType()=="Layer")
                {
                    dynamic_cast<Layer<T>*>(m_layers[i])->SetParameters(params[i]);
                }                
            }
            
        }
        
    };
} // namespace MiniBrain
