#pragma once
#include <autodiff/reverse/var/eigen.hpp>
#include "../Layer.h"
#include "../Optimizer.h"

namespace MiniBrain
{
    template<typename T>
    class GRU : public Layer<T>
    {
    protected:
        int m_hiddenSize;
        int m_BatchSize;

        Matrix<T> m_z;
        Matrix<T> m_r;
        // Matrix<T> m_h;

        Matrix<Scalar> m_h_prev;        

        Matrix<T> m_weight_z;
        Matrix<T> m_weight_r;
        Matrix<T> m_weight_h;

        Vector<T> m_bias_z;
        Vector<T> m_bias_r;
        Vector<T> m_bias_h;

        Matrix<T> m_Uz;
        Matrix<T> m_Ur;
        Matrix<T> m_Uh;
        
        Matrix<Scalar> m_dWz,m_dWr,m_dWh;
        Matrix<Scalar> m_dUz,m_dUr,m_dUh;
        Vector<Scalar> m_dbz,m_dbr,m_dbh;
        // Matrix<T> m_din;

        Matrix<T> Sigmoid(const Matrix<T>& InData)
        {
            return 1.0f/(1.0f + (-InData.array()).exp());
        }

        static void SerializeParameter(const Matrix<Scalar>& m, std::vector<Scalar>& params, int offset=0)
        {
            std::copy(m.data(),m.data()+static_cast<int>(m.size()),params.begin()+offset);
        }

        static void DeserializeParameter(const std::vector<Scalar>& params, Matrix<T>& m, int offset=0)
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                for (int i = 0; i < m.size(); i++)
                {
                    m.reshaped()(i) = params[offset + i];
                }
            }
            else
            {
                std::copy(params.begin()+offset,params.begin()+offset+static_cast<int>(m.size()),m.data());
            }
        }

        static void DeserializeParameter(const std::vector<Scalar>& params, Vector<T>& m, int offset=0)
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                for (int i = 0; i < m.size(); i++)
                {
                    m.reshaped()(i) = params[offset + i];
                }
            }
            else
            {
                std::copy(params.begin()+offset,params.begin()+offset+static_cast<int>(m.size()),m.data());
            }
        }

    public:
        GRU (int inSize, int hiddenSize):Layer<T>(inSize,hiddenSize),m_hiddenSize(hiddenSize)
        {
            m_BatchSize = 1;
            Init();
        }
        ~GRU () {}

        virtual Matrix<T> Forward(const Matrix<T>& InData) override
        {
            const int nobs = InData.cols();
            if (nobs != m_BatchSize)
            {
                SetBatchSize(nobs);
            }
            Matrix<T> m_h(m_hiddenSize, nobs);
            Matrix<T> m_h_tilde;
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                using autodiff::reverse::detail::tanh;

                Matrix<T> h_prev_T = m_h_prev.template cast<T>();
                m_z = Sigmoid((m_weight_z.transpose() * InData + m_Uz.transpose() * m_h_prev.cast<MiniBrain::AutoDiffVar>()).colwise()+m_bias_z);
                m_r = Sigmoid((m_weight_r.transpose() * InData + m_Ur.transpose() * m_h_prev.cast<MiniBrain::AutoDiffVar>()).colwise()+m_bias_r);
                
                Matrix<T> r_h(m_hiddenSize, nobs);
                for(int j = 0; j < nobs; ++j) {
                    r_h.col(j) = m_r.col(j).array() * h_prev_T.col(j).array();
                }
                Matrix<T> h_lin = (m_weight_h.transpose() * InData + m_Uh.transpose() * r_h).colwise() + m_bias_h;
                m_h_tilde = h_lin.unaryExpr([](const AutoDiffVar& x) -> AutoDiffVar { return tanh(x); });
                
                m_h.resize(m_hiddenSize, nobs);
                for(int j = 0; j < nobs; ++j) {
                    m_h.col(j) = (1.0f - m_z.col(j).array()) * h_prev_T.col(j).array() + m_z.col(j).array() * m_h_tilde.col(j).array();
                }

                // 提取数值，刷新常驻历史记忆
                m_h_prev = m_h.unaryExpr([](const T& x) {
                    return static_cast<Scalar>(x.expr->val);
                });
            }
            else
            {
                m_z.noalias() = Sigmoid((m_weight_z.transpose() * InData + m_Uz.transpose() * m_h_prev).colwise()+m_bias_z);
                m_r.noalias() = Sigmoid((m_weight_r.transpose() * InData + m_Ur.transpose() * m_h_prev).colwise()+m_bias_r);
                m_h_tilde.array() = ((m_weight_h.transpose() * InData + m_Uh.transpose() * (m_r.array() * m_h_prev.array()).matrix()).colwise() + m_bias_h).array().tanh();
                m_h.array() = (1.0f - m_z.array())*m_h_prev.array() + m_z.array()*m_h_tilde.array();
                
                // 提取数值，刷新常驻历史记忆
                m_h_prev = m_h;
            }

            return m_h;
        }

        virtual void Backward(T& Loss) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                int total_params = 
                    m_weight_z.size() + m_Uz.size() + m_bias_z.size() +
                    m_weight_r.size() + m_Ur.size() + m_bias_r.size() +
                    m_weight_h.size() + m_Uh.size() + m_bias_h.size();

                Vector<AutoDiffVar> params(total_params);
                // 按严格的固有内存顺序，把 9 个矩阵的大乱炖“全量展平拼接”
                int offset = 0;
                auto pack = [&](const auto& matrix) {
                    params.segment(offset, matrix.size()) = matrix.reshaped();
                    offset += matrix.size();
                };
                
                pack(m_weight_z); pack(m_Uz); pack(m_bias_z);
                pack(m_weight_r); pack(m_Ur); pack(m_bias_r);
                pack(m_weight_h); pack(m_Uh); pack(m_bias_h);

                Vector<Scalar> gradients = autodiff::gradient(Loss, params);
                // 按照同样的顺序，把梯度“全量展平拼接”回 9 个矩阵
                offset = 0;
                auto unpack = [&](auto& gradDest, const auto& matrix) {
                    gradDest = gradients.segment(offset, matrix.size()).reshaped(matrix.rows(), matrix.cols());
                    offset += matrix.size();
                };

                unpack(m_dWz, m_weight_z); unpack(m_dUz, m_Uz); unpack(m_dbz, m_bias_z);
                unpack(m_dWr, m_weight_r); unpack(m_dUr, m_Ur); unpack(m_dbr, m_bias_r);
                unpack(m_dWh, m_weight_h); unpack(m_dUh, m_Uh); unpack(m_dbh, m_bias_h);
            }
            /*const int nobs = InData.cols();
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
            m_dbz.noalias() = m_dz.rowwise().mean();*/

            // m_din.resize(m_inSize,nobs);
            // m_din.noalias() = m_weight_h * m_dh + m_weight_r * m_dr + m_weight_z * m_dz;
        }

        virtual void Init() override
        {
            m_weight_z.resize(this->m_inSize,this->m_hiddenSize);
            m_weight_r.resize(this->m_inSize,this->m_hiddenSize);
            m_weight_h.resize(this->m_inSize,this->m_hiddenSize);
            m_Uz.resize(this->m_hiddenSize,this->m_hiddenSize);
            m_Ur.resize(this->m_hiddenSize,this->m_hiddenSize);
            m_Uh.resize(this->m_hiddenSize,this->m_hiddenSize);
            m_bias_z.resize(this->m_hiddenSize);
            m_bias_r.resize(this->m_hiddenSize);
            m_bias_h.resize(this->m_hiddenSize);

            m_h_prev.resize(this->m_hiddenSize,1);
            // m_h_tilde.resize(this->m_hiddenSize,1);
            // m_h.resize(this->m_hiddenSize,1);

            m_dWz.resize(this->m_inSize,this->m_hiddenSize);
            m_dWr.resize(this->m_inSize,this->m_hiddenSize);
            m_dWh.resize(this->m_inSize,this->m_hiddenSize);
            m_dUz.resize(this->m_hiddenSize,this->m_hiddenSize);
            m_dUr.resize(this->m_hiddenSize,this->m_hiddenSize);
            m_dUh.resize(this->m_hiddenSize,this->m_hiddenSize);
            m_dbz.resize(this->m_hiddenSize);
            m_dbr.resize(this->m_hiddenSize);
            m_dbh.resize(this->m_hiddenSize);
        }

        virtual void Init(const Scalar& mu, const Scalar& sigma, Random& RNG) override
        {
            Init();
            RNG.SetNormalDistRandom(m_weight_z,mu,sigma);
            RNG.SetNormalDistRandom(m_weight_r,mu,sigma);
            RNG.SetNormalDistRandom(m_weight_h,mu,sigma);
            RNG.SetNormalDistRandom(m_Uz,mu,sigma);
            RNG.SetNormalDistRandom(m_Ur,mu,sigma);
            RNG.SetNormalDistRandom(m_Uh,mu,sigma);
        }

        virtual void Update(Optimizer<Scalar>& opt) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                opt.Update(m_dWz, m_weight_z);
                opt.Update(m_dWr, m_weight_r);
                opt.Update(m_dWh, m_weight_h);
                opt.Update(m_dUz, m_Uz);
                opt.Update(m_dUr, m_Ur);
                opt.Update(m_dUh, m_Uh);
                opt.Update(m_dbz, m_bias_z);
                opt.Update(m_dbr, m_bias_r);
                opt.Update(m_dbh, m_bias_h);
            }
        }

        void SetBatchSize(int Size)
        {
            m_BatchSize = Size;
            //隐状态数据比较特殊，不能随意改变batch大小，否则会导致数据丢失
            // m_h.resize(m_hiddenSize, Size);
            m_h_prev.resize(this->m_hiddenSize, Size);
            // m_h_tilde.resize(this->m_hiddenSize, Size);

            m_z.resize(this->m_hiddenSize,Size);
            m_r.resize(this->m_hiddenSize,Size);
        }

        void ResetAllMemory()
        {
            // m_h.setZero();
            m_h_prev.setZero();
            // m_h_tilde.setZero();
        }

        void ResetMemory(int index)
        {
           m_h_prev.col(index).setZero();
        }

        virtual std::vector<Scalar> GetParameters() const override
        {
            int size = m_weight_z.size()+m_weight_r.size()+m_weight_h.size()+
            m_Uz.size()+m_Ur.size()+m_Uh.size()+
            m_bias_z.size()+m_bias_r.size()+m_bias_h.size();
            std::vector<Scalar> params(size);
            
            const Matrix<Scalar>* z;
            const Matrix<Scalar>* r;
            const Matrix<Scalar>* h;

            const Vector<Scalar>* bias_z;
            const Vector<Scalar>* bias_r;
            const Vector<Scalar>* bias_h;

            const Matrix<Scalar>* Uz;
            const Matrix<Scalar>* Ur;
            const Matrix<Scalar>* Uh;

            Matrix<Scalar> z_cache;
            Matrix<Scalar> r_cache;
            Matrix<Scalar> h_cache;

            Vector<Scalar> bias_z_cache;
            Vector<Scalar> bias_r_cache;
            Vector<Scalar> bias_h_cache;

            Matrix<Scalar> Uz_cache;
            Matrix<Scalar> Ur_cache;
            Matrix<Scalar> Uh_cache;

            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                z_cache = m_weight_z.unaryExpr([](const AutoDiffVar& x) { return static_cast<Scalar>(x.expr->val); });
                r_cache = m_weight_r.unaryExpr([](const AutoDiffVar& x) { return static_cast<Scalar>(x.expr->val); });
                h_cache = m_weight_h.unaryExpr([](const AutoDiffVar& x) { return static_cast<Scalar>(x.expr->val); });
                bias_z_cache = m_bias_z.unaryExpr([](const AutoDiffVar& x) { return static_cast<Scalar>(x.expr->val); });
                bias_r_cache = m_bias_r.unaryExpr([](const AutoDiffVar& x) { return static_cast<Scalar>(x.expr->val); });
                bias_h_cache = m_bias_h.unaryExpr([](const AutoDiffVar& x) { return static_cast<Scalar>(x.expr->val); });
                Uz_cache = m_Uz.unaryExpr([](const AutoDiffVar& x) { return static_cast<Scalar>(x.expr->val); });
                Ur_cache = m_Ur.unaryExpr([](const AutoDiffVar& x) { return static_cast<Scalar>(x.expr->val); });
                Uh_cache = m_Uh.unaryExpr([](const AutoDiffVar& x) { return static_cast<Scalar>(x.expr->val); });

                z = &z_cache;
                r = &r_cache;
                h = &h_cache;
                bias_z = &bias_z_cache;
                bias_r = &bias_r_cache;
                bias_h = &bias_h_cache;
                Uz = &Uz_cache;
                Ur = &Ur_cache;
                Uh = &Uh_cache;
            }
            else
            {
                z = &m_weight_z;
                r = &m_weight_r;
                h = &m_weight_h;
                bias_z = &m_bias_z;
                bias_r = &m_bias_r;
                bias_h = &m_bias_h;
                Uz = &m_Uz;
                Ur = &m_Ur;
                Uh = &m_Uh;
            }

            int offset=0;
            SerializeParameter(*z,params,offset);
            offset+=m_weight_z.size();
            SerializeParameter(*r,params,offset);
            offset+=m_weight_r.size();
            SerializeParameter(*h,params,offset);
            offset+=m_weight_h.size();

            SerializeParameter(*Uz,params,offset);
            offset+=m_Uz.size();
            SerializeParameter(*Ur,params,offset);
            offset+=m_Ur.size();
            SerializeParameter(*Uh,params,offset);
            offset+=m_Uh.size();

            SerializeParameter(*bias_z,params,offset);
            offset+=m_bias_z.size();
            SerializeParameter(*bias_r,params,offset);
            offset+=m_bias_r.size();
            SerializeParameter(*bias_h,params,offset);
            offset+=m_bias_h.size();

            return params;
        }

        virtual void SetParameters(const std::vector<Scalar>& param) override
        {
            int size = m_weight_z.size()+m_weight_r.size()+m_weight_h.size()+
            m_Uz.size()+m_Ur.size()+m_Uh.size()+
            m_bias_z.size()+m_bias_r.size()+m_bias_h.size();
            if (static_cast<int>(param.size())!=size)
            {
                MINIBRAIN_THROW(std::invalid_argument("GRU: parameter size mismatch"));
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
