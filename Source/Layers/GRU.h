#pragma once
#include "../Layer.h"
#include "../Optimizer.h"

namespace MiniBrain
{
    class GRU : public Layer
    {
    protected:
        int m_hiddenSize;
        int m_BatchSize;

        Matrix m_z;
        Matrix m_r;
        Matrix m_h;

        Matrix m_h_prev;
        Matrix m_h_tilde;

        Matrix m_weight_z;
        Matrix m_weight_r;
        Matrix m_weight_h;

        Vector m_bias_z;
        Vector m_bias_r;
        Vector m_bias_h;

        Matrix m_Uz;
        Matrix m_Ur;
        Matrix m_Uh;
        
        Matrix m_dWz,m_dWr,m_dWh;
        Matrix m_dUz,m_dUr,m_dUh;
        Vector m_dbz,m_dbr,m_dbh;
        Matrix m_din;

        Matrix Sigmoid(const Matrix& InData)
        {
            return 1.0f/(1.0f + (-InData.array()).exp());
        }

        static void SerializeParameter(const Matrix& m, std::vector<float>& params, int offset=0)
        {
            std::copy(m.data(),m.data()+static_cast<int>(m.size()),params.begin()+offset);
        }

        static void DeserializeParameter(const std::vector<float>& params, Matrix& m, int offset=0)
        {
            std::copy(params.begin()+offset,params.begin()+offset+static_cast<int>(m.size()),m.data());
        }

        static void DeserializeParameter(const std::vector<float>& params, Vector& m, int offset=0)
        {
            std::copy(params.begin()+offset,params.begin()+offset+static_cast<int>(m.size()),m.data());
        }

    public:
        GRU (int inSize, int hiddenSize):Layer(inSize,hiddenSize),m_hiddenSize(hiddenSize)
        {
            m_BatchSize = 1;
        }
        ~GRU () {}

        virtual void Forward(const Matrix& InData) override
        {
            const int nobs = InData.cols();
            if (nobs != m_BatchSize)
            {
                SetBatchSize(nobs);
            }
            
            m_h_prev = m_h;
            m_z.noalias() = Sigmoid((m_weight_z.transpose() * InData + m_Uz.transpose() * m_h_prev).colwise()+m_bias_z);
            m_r.noalias() = Sigmoid((m_weight_r.transpose() * InData + m_Ur.transpose() * m_h_prev).colwise()+m_bias_r);
            m_h_tilde.array() = ((m_weight_h.transpose() * InData + m_Uh.transpose() * (m_r.array() * m_h_prev.array()).matrix()).colwise() + m_bias_h).array().tanh();
            m_h.array() = (1.0f - m_z.array())*m_h_prev.array() + m_z.array()*m_h_tilde.array();
        }

        virtual void Backward(const Matrix& InData, const Matrix& BackpropData) override
        {
            const int nobs = InData.cols();
            Matrix m_dz(m_hiddenSize,nobs),m_dr(m_hiddenSize,nobs),m_dh(m_hiddenSize,nobs);
            m_dz.array() = BackpropData.array() * (m_h_tilde.array() - m_h.array()) * m_z.array() * (1.0f-m_z.array());
            m_dh.array() = BackpropData.array() * m_z.array() * (1.0f-m_h_tilde.array().square());
            m_dr.array() = (m_Uh*m_dh).array()*m_h.array()*m_r.array()*(1.0f-m_r.array());
            
            //dot(x.T, dh)
            m_dWh.noalias() = m_dh*InData.transpose();
            //dot(r*h).T*dh
            m_dUh.noalias() = m_dh*(m_r.array()*m_h.array()).matrix().transpose();
            m_dbh.noalias() = m_dh.rowwise().mean();

            m_dWr.noalias() = m_dr*InData.transpose();
            //dot(m_h.T, m_dr) 
            m_dUr.noalias() = m_dr*m_h.transpose();
            m_dbr.noalias() = m_dr.rowwise().mean();

            m_dWz.noalias() = m_dz*InData.transpose();
            //dot(m_h.T, m_dz)
            m_dUz.noalias() = m_dz*m_h.transpose();
            m_dbz.noalias() = m_dz.rowwise().mean();

            m_din.resize(m_inSize,nobs);
            m_din.noalias() = m_weight_h * m_dh + m_weight_r * m_dr + m_weight_z * m_dz;
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
            m_weight_z.resize(m_inSize,m_hiddenSize);
            m_weight_r.resize(m_inSize,m_hiddenSize);
            m_weight_h.resize(m_inSize,m_hiddenSize);
            m_Uz.resize(m_hiddenSize,m_hiddenSize);
            m_Ur.resize(m_hiddenSize,m_hiddenSize);
            m_Uh.resize(m_hiddenSize,m_hiddenSize);
            m_bias_z.resize(m_hiddenSize);
            m_bias_r.resize(m_hiddenSize);
            m_bias_h.resize(m_hiddenSize);

            m_h_prev.resize(m_hiddenSize,1);
            m_h_tilde.resize(m_hiddenSize,1);
            m_h.resize(m_hiddenSize,1);

            m_dWz.resize(m_inSize,m_hiddenSize);
            m_dWr.resize(m_inSize,m_hiddenSize);
            m_dWh.resize(m_inSize,m_hiddenSize);
            m_dUz.resize(m_hiddenSize,m_hiddenSize);
            m_dUr.resize(m_hiddenSize,m_hiddenSize);
            m_dUh.resize(m_hiddenSize,m_hiddenSize);
            m_dbz.resize(m_hiddenSize);
            m_dbr.resize(m_hiddenSize);
            m_dbh.resize(m_hiddenSize);
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
        }

        virtual void Update(Optimizer& opt) override
        {
            AlignedMapVec Wz(m_weight_z.data(), m_weight_z.size());
            AlignedMapVec Wr(m_weight_r.data(), m_weight_r.size());
            AlignedMapVec Wh(m_weight_h.data(), m_weight_h.size());
            AlignedMapVec Uz(m_Uz.data(), m_Uz.size());
            AlignedMapVec Ur(m_Ur.data(), m_Ur.size());
            AlignedMapVec Uh(m_Uh.data(), m_Uh.size());
            AlignedMapVec Bz(m_bias_z.data(), m_bias_z.size());
            AlignedMapVec Br(m_bias_r.data(), m_bias_r.size());
            AlignedMapVec Bh(m_bias_z.data(), m_bias_z.size());
            opt.Update(ConstAlignedMapVec(m_dWz.data(), m_dWz.size()), Wz);
            opt.Update(ConstAlignedMapVec(m_dWr.data(), m_dWr.size()), Wr);
            opt.Update(ConstAlignedMapVec(m_dWh.data(), m_dWh.size()), Wh);
            opt.Update(ConstAlignedMapVec(m_dUz.data(), m_dUz.size()), Uz);
            opt.Update(ConstAlignedMapVec(m_dUr.data(), m_dUr.size()), Ur);
            opt.Update(ConstAlignedMapVec(m_dUh.data(), m_dUh.size()), Uh);
            opt.Update(ConstAlignedMapVec(m_dbz.data(), m_dbz.size()), Bz);
            opt.Update(ConstAlignedMapVec(m_dbr.data(), m_dbr.size()), Br);
            opt.Update(ConstAlignedMapVec(m_dbh.data(), m_dbh.size()), Bh);
        }

        void SetBatchSize(int Size)
        {
            m_BatchSize = Size;
            //隐状态数据比较特殊，不能随意改变batch大小，否则会导致数据丢失
            m_h.resize(m_hiddenSize, Size);
            m_h_prev.resize(m_hiddenSize, Size);
            m_h_tilde.resize(m_hiddenSize, Size);

            m_z.resize(m_hiddenSize,Size);
            m_r.resize(m_hiddenSize,Size);
        }

        void ResetMemory()
        {
            m_h.setZero();
            m_h_prev.setZero();
            m_h_tilde.setZero();
        }

        virtual std::vector<float> GetParameters() const override
        {
            int size = m_weight_z.size()+m_weight_r.size()+m_weight_h.size()+
            m_Uz.size()+m_Ur.size()+m_Uh.size()+
            m_bias_z.size()+m_bias_r.size()+m_bias_h.size();
            std::vector<float> params(size);

            int offset=0;
            SerializeParameter(m_weight_z,params,offset);
            offset+=m_weight_z.size();
            SerializeParameter(m_weight_r,params,offset);
            offset+=m_weight_r.size();
            SerializeParameter(m_weight_h,params,offset);
            offset+=m_weight_h.size();

            SerializeParameter(m_Uz,params,offset);
            offset+=m_Uz.size();
            SerializeParameter(m_Ur,params,offset);
            offset+=m_Ur.size();
            SerializeParameter(m_Uh,params,offset);
            offset+=m_Uh.size();

            SerializeParameter(m_bias_z,params,offset);
            offset+=m_bias_z.size();
            SerializeParameter(m_bias_r,params,offset);
            offset+=m_bias_r.size();
            SerializeParameter(m_bias_h,params,offset);
            offset+=m_bias_h.size();

            return params;
        }

        virtual void SetParameters(const std::vector<float>& param) override
        {
            int size = m_weight_z.size()+m_weight_r.size()+m_weight_h.size()+
            m_Uz.size()+m_Ur.size()+m_Uh.size()+
            m_bias_z.size()+m_bias_r.size()+m_bias_h.size();
            if (static_cast<int>(param.size())!=size)
            {
                throw std::invalid_argument("GRU: parameter size mismatch");
            }
            int offset = 0;
            DeserializeParameter(param,m_weight_z,offset);
            offset += m_weight_z.size();
            DeserializeParameter(param,m_weight_r,offset);
            offset += m_weight_r.size();
            DeserializeParameter(param,m_weight_h,offset);
            offset += m_weight_h.size();

            DeserializeParameter(param,m_Uz,offset);
            offset += m_Uz.size();
            DeserializeParameter(param,m_Ur,offset);
            offset += m_Ur.size();
            DeserializeParameter(param,m_Uh,offset);
            offset += m_Uh.size();

            DeserializeParameter(param,m_bias_z,offset);
            offset += m_bias_z.size();
            DeserializeParameter(param,m_bias_r,offset);
            offset += m_bias_r.size();
            DeserializeParameter(param,m_bias_h,offset);
            offset += m_bias_h.size();
        }

        virtual std::string GetSubType() const override
        {
            return "GRU";
        }
    };
} // namespace MiniBrain
