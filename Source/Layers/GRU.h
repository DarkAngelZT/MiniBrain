#pragma once
#include "../Layer.h"
#include "../Optimizer.h"

namespace MiniBrain
{
    class GRU : public Layer
    {
    protected:
        Matrix m_z;
        Matrix m_r;
        Matrix m_h;

        Matrix m_h_prev;
        Matrix m_h_tilde;

        Matrix m_weight_z;
        Matrix m_weight_r;
        Matrix m_weight_h;

        Matrix m_Uz;
        Matrix m_Ur;
        Matrix m_Uh;

        Vector m_bias_z;
        Vector m_bias_r;
        Vector m_bias_h;

        Matrix m_dz,m_dr,m_dh;
        Matrix m_dWz,m_dWr,m_dWh;
        Matrix m_dUz,m_dUr,m_dUh;
        Matrix m_din;

        Matrix Sigmoid(const Matrix& InData)
        {
            return 1.0f/(1.0f + (-InData.array()).exp());
        }
    public:
        GRU (int inSize,int OutSize):Layer(inSize,OutSize)
        {

        }
        ~GRU () {}

        virtual void Forward(const Matrix& InData) override
        {
            const int nobs = InData.cols();
            //out = w .* in + b
            m_h.resize(m_outSize, nobs);
            m_h_prev = m_h;
            m_z.noalias() = Sigmoid(m_weight_z.transpose() * InData + m_Uz.transpose() * m_h_prev + m_bias_z);
            m_r.noalias() = Sigmoid(m_weight_r.transpose() * InData + m_Ur.transpose() * m_h_prev + m_bias_r);
            m_h_tilde.array() = (m_weight_h.transpose() * InData + m_Ur.transpose() * (m_r.array() * m_h_prev.array()).matrix() + m_bias_h).array().tanh();
            m_h.array() = (1.0f - m_z.array())*m_h_prev.array() + m_z.array()*m_h_tilde.array();
        }

        virtual void Backward(const Matrix& InData, const Matrix& BackpropData) override
        {
            m_dz.array() = BackpropData.array() * (m_h_tilde.array() - m_h.array()) * m_z.array() * (1.0f-m_z.array());
            m_dh.array() = BackpropData.array() * m_z.array() * (1.0f-m_h_tilde.array().square());
            m_dr.array() = (m_weight_h.transpose()*m_dh).array()*m_h.array()*m_r.array()*(1.0f-m_r.array());

            m_dWh = m_dh.transpose() * InData.transpose();
            m_dUh = m_dh.transpose() * (m_r.array()*m_h.array()).matrix().transpose();
            m_dWr = m_dr.transpose() * InData.transpose();
            m_dUr = m_dr.transpose() * m_h.transpose();
            m_dWz = m_dz.transpose() * InData.transpose();
            m_dUz = m_dz.transpose() * m_h.transpose();

            //dx = dot(uh.T,dh) + dot(ur.T,dr) + dot(uz.T, dz)
            //dot(Uh.t,dh) = Uh.T.T * dh
            m_din = m_Uh * m_dh + m_Ur * m_dr + m_Uz * m_dz;
        }

        virtual const Matrix& Output() const override
        {
            return m_h;
        }

        virtual const Matrix& GetBackpropData() const override
        {
            return m_din;
        }

        virtual void Init() override
        {
            m_weight_z.resize(m_inSize,m_outSize);
            m_weight_r.resize(m_inSize,m_outSize);
            m_weight_h.resize(m_inSize,m_outSize);
            m_Uz.resize(m_outSize,m_outSize);
            m_Ur.resize(m_outSize,m_outSize);
            m_Uh.resize(m_outSize,m_outSize);
            m_bias_z.resize(m_outSize);
            m_bias_r.resize(m_outSize);
            m_bias_h.resize(m_outSize);
            m_h_prev.resize(m_outSize,1);
            m_dWz.resize(m_inSize,m_outSize);
            m_dWr.resize(m_inSize,m_outSize);
            m_dWh.resize(m_inSize,m_outSize);
            m_dUz.resize(m_outSize,m_outSize);
            m_dUr.resize(m_outSize,m_outSize);
            m_dUz.resize(m_outSize,m_outSize);
            m_din.resize(m_inSize,1);
        }

        virtual void Init(const float& mu, const float& sigma, Random& RNG) override
        {
            Init();
            RNG.SetNormalDistRandom(m_weight_z.data(),m_weight_z.size(),mu,sigma);
            RNG.SetNormalDistRandom(m_weight_r.data(),m_weight_r.size(),mu,sigma);
            RNG.SetNormalDistRandom(m_weight_h.data(),m_weight_h.size(),mu,sigma);
            RNG.SetNormalDistRandom(m_Uz.data(),m_Uz.size(),mu,sigma);
            RNG.SetNormalDistRandom(m_Ur.data(),m_Ur.size(),mu,sigma);
            RNG.SetNormalDistRandom(m_Uh.data(),m_Uh.size(),mu,sigma);
            RNG.SetNormalDistRandom(m_bias_z.data(),m_bias_z.size(),mu,sigma);
            RNG.SetNormalDistRandom(m_bias_r.data(),m_bias_r.size(),mu,sigma);
            RNG.SetNormalDistRandom(m_bias_h.data(),m_bias_h.size(),mu,sigma);
        }

        virtual void Update(Optimizer& opt) override
        {
            AlignedMapVec Wz(m_weight_z.data(), m_weight_z.size());
            AlignedMapVec Wr(m_weight_r.data(), m_weight_r.size());
            AlignedMapVec Wh(m_weight_h.data(), m_weight_h.size());
            AlignedMapVec Uz(m_Uz.data(), m_Uz.size());
            AlignedMapVec Ur(m_Ur.data(), m_Ur.size());
            AlignedMapVec Uh(m_Uh.data(), m_Uh.size());
            opt.Update(ConstAlignedMapVec(m_dWz.data(), m_dWz.size()), Wz);
            opt.Update(ConstAlignedMapVec(m_dWr.data(), m_dWr.size()), Wr);
            opt.Update(ConstAlignedMapVec(m_dWh.data(), m_dWh.size()), Wh);
            opt.Update(ConstAlignedMapVec(m_dUz.data(), m_dUz.size()), Uz);
            opt.Update(ConstAlignedMapVec(m_dUr.data(), m_dUr.size()), Ur);
            opt.Update(ConstAlignedMapVec(m_dUh.data(), m_dUh.size()), Uh);
            // opt.Update(ConstAlignedMapVec(m_dbias_z.data(), m_dbias_z.size()), ConstAlignedMapVec(m_bias_z.data(), m_bias_z.size()));
            // opt.Update(ConstAlignedMapVec(m_dbias_r.data(), m_dbias_r.size()), ConstAlignedMapVec(m_bias_r.data(), m_bias_r.size()));
            // opt.Update(ConstAlignedMapVec(m_dbias_h.data(), m_dbias_h.size()), ConstAlignedMapVec(m_bias_h.data(), m_bias_h.size()));
        }

        virtual std::string GetSubType() const override
        {
            return "GRU";
        }
    };
} // namespace MiniBrain
