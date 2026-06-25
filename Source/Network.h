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
        std::vector<std::unique_ptr<IComputeNode<T>>> m_layers;
        std::unique_ptr<LossFunc> m_lossFunc=nullptr;
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
                if (static_cast<Layer<T>*>(m_layers[i].get())->GetInSize() != static_cast<Layer<T>*>(m_layers[j].get())->GetOutSize())
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
                return Matrix<T>();
            }
            
            if(InData.rows() != static_cast<Layer<T>*>(m_layers[0].get())->GetInSize())
            {
                MINIBRAIN_THROW(std::invalid_argument("[Network]: Input data dimension mismatch"));
            }

            Matrix<T> output = m_layers[0]->Forward(InData);

            for (int i = 1; i < numLayers; i++)
            {
                output = m_layers[i]->Forward(output);
            }
            return output;
        }

        virtual void Backward(const Matrix<T>& Output, const Matrix<T>& Target)
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                
                const int nLayer = GetLayerAmount();
                if (nLayer <= 0)
                {
                    return;
                }

                AutoDiffVar loss = m_lossFunc->Evaluate(Output, Target);

                Backward(loss);
            }
        }

        virtual void Backward(AutoDiffVar& loss)
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                const int nLayer = GetLayerAmount();
                if (nLayer <= 0)
                {
                    return;
                }

                for (int i = nLayer-1; i >=0; i--)
                {
                    m_layers[i]->Backward(loss);
                }
            }
        }

        virtual void Update(Optimizer<Scalar>& opt)
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
                    dynamic_cast<Layer<T>*>(m_layers[i].get())->Update(opt);
                }                
            } 
        }

        Network() :
            m_lossFunc(nullptr),
            m_rng(m_defaultRNG)
        {}
        ~Network() 
        {
            m_layers.clear();
            m_lossFunc.reset();
        }

        virtual void Init(const Scalar& mu=0.f, const Scalar& sigma=1.f)
        {
            if(!CheckUnitSize())
            {
                MINIBRAIN_THROW(std::invalid_argument("[Network]Layer size mismatch"));
            }
            const int nLayers = GetLayerAmount();
            
            for (int i = 0; i < nLayers; i++)
            {
                if (m_layers[i]->GetType()=="Layer")
                {
                    dynamic_cast<Layer<T>*>(m_layers[i].get())->Init(mu,sigma,m_rng);
                }                
            }
            
        }

        void AddLayer(std::unique_ptr<IComputeNode<T>> layer)
        {
            m_layers.push_back(std::move(layer));
        }

        void SetLossFunc(std::unique_ptr<LossFunc> lossFunc)
        {
            m_lossFunc = std::move(lossFunc);
        }

        int GetLayerAmount() const
        {
            return m_layers.size();
        }

        const LossFunc* GetLossFunc() const
        {
            return m_lossFunc.get();
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
                    auto LayerPtr = dynamic_cast<Layer<T>*>(m_layers[i].get());
                    if (LayerPtr and LayerPtr->HasParameters())
                    {
                        result.push_back(LayerPtr->GetParameters());
                    }
                }               
            }            

            return result;
        }

        virtual void SetParameters(const std::vector<std::vector<Scalar>>& params)
        {
            const int nLayer = GetLayerAmount();
            if (static_cast<int>(params.size())!=nLayer)
            {
                MINIBRAIN_THROW(std::invalid_argument("[Network]: parameter size mismatch"));
            }
            
            for (int i = 0; i < nLayer; i++)
            {
                if (m_layers[i]->GetType()=="Layer")
                {
                    auto LayerPtr = dynamic_cast<Layer<T>*>(m_layers[i].get());
                    if (LayerPtr and LayerPtr->HasParameters())
                    {
                        LayerPtr->SetParameters(params[i]);
                    }
                }                
            }
            
        }
        
    };
} // namespace MiniBrain
