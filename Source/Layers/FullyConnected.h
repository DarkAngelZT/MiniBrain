#pragma once
#include <autodiff/reverse/var/eigen.hpp>
#include "../Eigen/Dense"
#include "../Layer.h"
#include "../Optimizer.h"
#include <vector>

namespace MiniBrain
{
    template<typename T>
    class FullyConnected: public Layer<T>
    {        
    protected:
        Matrix<T> m_weight;
        Vector<T> m_bias;

        //weight的导数
        Matrix<Scalar> m_dw;
        //bias的导数
        Vector<Scalar> m_db;

        //合并格式，z=w*x+b,当前层的输出
        // Matrix<T> m_out;
        //输入端的反向传播输出
        // Matrix<T> m_din;

    public:
        FullyConnected(int inSize,int OutSize):Layer<T>(inSize,OutSize)
        {
            Init();
        }

        virtual void Init() override
        {
            m_weight.resize(this->m_inSize,this->m_outSize);
            m_bias.resize(this->m_outSize);
            m_dw.resize(this->m_inSize,this->m_outSize);
            m_db.resize(this->m_outSize);
        }

        virtual void Init(const Scalar& mu, const Scalar& sigma, Random& RNG) override
        {
            Init();
            RNG.SetNormalDistRandom(m_weight,mu,sigma);
            RNG.SetNormalDistRandom(m_bias,mu,sigma);
        }

        virtual Matrix<T> Forward(const Matrix<T>& InData) override
        {
            const int nobs = InData.cols();
            //out = w .* in + b
            // m_out.resize(m_outSize, nobs);
            Matrix<T> m_out(this->m_outSize, nobs);
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                m_out = m_weight.transpose()*InData;
                m_out.colwise() += m_bias;
            }
            else
            {
                m_out.noalias() = m_weight.transpose()*InData;
                m_out.colwise() += m_bias;
            }
            return m_out;
        }

        virtual void Backward(T& Loss) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                m_dw.setZero();
                m_db.setZero();
                Vector<AutoDiffVar> params(m_weight.size()+m_bias.size());
                params<< m_weight.reshaped(), m_bias.reshaped();
                Eigen::VectorXf grads = autodiff::gradient(Loss,params);
                m_dw = grads.head(m_weight.size()).reshaped(this->m_inSize,this->m_outSize);
                m_db = grads.tail(m_bias.size());
            }
            // const int nobs = LastLayerData.cols();
            // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
            // m_dw.noalias() = LastLayerData * NextLayerData.transpose() / nobs;
            // Derivative for bias, d(L) / d(b) = d(L) / d(z)
            // m_db.noalias() = NextLayerData.rowwise().mean();
            // Compute d(L) / d_in = W * [d(L) / d(z)]
            // m_din.resize(m_inSize,nobs);
            // m_din.noalias() = m_weight * NextLayerData;
        }

        virtual void Update(Optimizer<Scalar>& opt) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                opt.Update(m_dw, m_weight);
                opt.Update(m_db, m_bias);
            }            
        }

        virtual int GetAutoDiffParameterCount() const override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
                return static_cast<int>(m_weight.size() + m_bias.size());
            return 0;
        }

        virtual void AppendAutoDiffParameters(
            Vector<AutoDiffVar>& destination,
            int& offset) const override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                const int weight_size = static_cast<int>(m_weight.size());
                const int bias_size = static_cast<int>(m_bias.size());
                const int count = weight_size + bias_size;
                if(offset < 0 || offset + count > destination.size())
                    MINIBRAIN_THROW(std::out_of_range("FullyConnected: parameter destination is too small"));

                // Keep the same stable order used by GetParameters/SetParameters:
                // all column-major weight elements first, followed by the bias.
                  // AutoDiffVar's copy constructor deliberately creates a new
                  // dependent-variable wrapper. Eigen's bulk assignment may
                  // therefore bind the gradient to that wrapper rather than to
                  // the parameter node used by Forward. Copy ExprPtr directly
                  // so every entry aliases the exact node in the graph.
                  for(int i = 0; i < weight_size; ++i)
                      destination(offset++).expr = m_weight.data()[i].expr;
                  for(int i = 0; i < bias_size; ++i)
                      destination(offset++).expr = m_bias.data()[i].expr;
            }
        }

        virtual void AssignGradients(
            const Vector<Scalar>& gradients,
            int& offset) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                const int weight_size = static_cast<int>(m_weight.size());
                const int bias_size = static_cast<int>(m_bias.size());
                const int count = weight_size + bias_size;
                if(offset < 0 || offset + count > gradients.size())
                    MINIBRAIN_THROW(std::out_of_range("FullyConnected: gradient source is too small"));

                m_dw = gradients.segment(offset, weight_size).reshaped(
                    m_weight.rows(), m_weight.cols());
                offset += weight_size;
                m_db = gradients.segment(offset, bias_size);
                offset += bias_size;
            }
        }

        virtual std::vector<Scalar> GetParameters() const override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                Vector<Scalar> w(m_weight.size());
                Vector<Scalar> b(m_bias.size());
                w = m_weight.reshaped().unaryExpr([](const AutoDiffVar& x){ return x.expr->val; });
                b = m_bias.reshaped().unaryExpr([](const AutoDiffVar& x){ return x.expr->val; });
                std::vector<Scalar> params(m_weight.size()+m_bias.size());
                std::copy(w.data(), w.data()+w.size(), params.begin());
                std::copy(b.data(), b.data()+b.size(), params.begin()+w.size());
                return params;
            }
            else
            {
                std::vector<Scalar> params(m_weight.size()+m_bias.size());
                std::copy(m_weight.data(),m_weight.data()+static_cast<int>(m_weight.size()),params.begin());
                std::copy(m_bias.data(),m_bias.data()+static_cast<int>(m_bias.size()),params.begin()+m_weight.size());
                return params;
            }
        }

        virtual void SetParameters(const std::vector<Scalar>& param) override
        {
            if (static_cast<int>(param.size())!=m_weight.size()+m_bias.size())
            {
                MINIBRAIN_THROW(std::invalid_argument("FullyConnected: parameter size mismatch"));
            }
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                for (int i = 0; i < m_weight.size(); i++)
                {
                    m_weight.reshaped()(i) = param[i];
                }
                for (int i = 0; i < m_bias.size(); i++)
                {
                    m_bias(i) = param[m_weight.size()+i];
                }
            }
            else
            {
                std::copy(param.begin(),param.begin()+static_cast<int>(m_weight.size()),m_weight.data());
                std::copy(param.begin()+static_cast<int>(m_weight.size()),param.end(),m_bias.data());
            }
        }

        virtual std::string GetSubType()const override{return "FullyConnected";}
    };
}
