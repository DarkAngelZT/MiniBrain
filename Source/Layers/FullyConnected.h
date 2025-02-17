#pragma once
#include "../Eigen/Dense"
#include "../Layer.h"
#include "../Optimizer.h"
#include <vector>

namespace MiniBrain
{
    class FullyConnected: public Layer
    {        
        Matrix m_weight;
        Vector m_bias;

        //weight的导数
        Matrix m_dw;
        //bias的导数
        Vector m_db;

        //合并格式，z=w*x+b,当前层的输出
        Matrix m_out;
        //输入端的反向传播输出
        Matrix m_din;

    public:
        FullyConnected(int inSize,int OutSize):Layer(inSize,OutSize)
        {}

        virtual const Matrix& Output() const override
        {
            return m_out;
        }

        virtual const Matrix& GetBackpropData() const override
        {
            return m_din;
        }

        virtual void Init() override
        {
            m_weight.resize(m_inSize,m_outSize);
            m_bias.resize(m_outSize);
            m_dw.resize(m_inSize,m_outSize);
            m_db.resize(m_outSize);
        }

        virtual void Init(const float& mu, const float& sigma, Random& RNG) override
        {
            Init();
            RNG.SetNormalDistRandom(m_weight.data(),m_weight.size(),mu,sigma);
            RNG.SetNormalDistRandom(m_bias.data(),m_bias.size(),mu,sigma);
        }

        virtual void Forward(const Matrix& InData) override
        {
            const int nobs = InData.cols();
            //out = w .* in + b
            m_out.resize(m_outSize, nobs);
            m_out.noalias() = m_weight.transpose()*InData;
            m_out.colwise() += m_bias;
        }

        virtual void Backward(const Matrix& LastLayerData,const Matrix& NextLayerData) override
        {
            const int nobs = LastLayerData.cols();
            // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
            m_dw.noalias() = LastLayerData * NextLayerData.transpose() / nobs;
            // Derivative for bias, d(L) / d(b) = d(L) / d(z)
            m_db.noalias() = NextLayerData.rowwise().mean();
            // Compute d(L) / d_in = W * [d(L) / d(z)]
            m_din.resize(m_inSize,nobs);
            m_din.noalias() = m_weight * NextLayerData;
        }

        virtual void Update(Optimizer&opt) override
        {
            ConstAlignedMapVec dw(m_dw.data(),m_dw.size());
            ConstAlignedMapVec db(m_db.data(),m_db.size());
            AlignedMapVec weight(m_weight.data(), m_weight.size());
            AlignedMapVec bias(m_bias.data(),m_bias.size());
            opt.Update(dw, weight);
            opt.Update(db, bias);
        }

        virtual std::vector<float> GetParameters() const override
        {
            std::vector<float> params(m_weight.size()+m_bias.size());
            std::copy(m_weight.data(),m_weight.data()+static_cast<int>(m_weight.size()),params.begin());
            std::copy(m_bias.data(),m_bias.data()+static_cast<int>(m_bias.size()),params.begin()+m_weight.size());
            return params;
        }

        virtual void SetParameters(const std::vector<float>& param) override
        {
            if (static_cast<int>(param.size())!=m_weight.size()+m_bias.size())
            {
                throw std::invalid_argument("FullyConnected: parameter size mismatch");
            }
            std::copy(param.begin(),param.begin()+static_cast<int>(m_weight.size()),m_weight.data());
            std::copy(param.begin()+static_cast<int>(m_weight.size()),param.end(),m_bias.data());
        }

        virtual std::string GetSubType()const override{return "FullyConnected";}
    };
}