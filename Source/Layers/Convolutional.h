#pragma once
#include <autodiff/reverse/var/eigen.hpp>
#include "../Eigen/Dense"
#include "../Layer.h"
#include "../Utils/Convolution.h"
#include "../Optimizer.h"

namespace MiniBrain
{
    template<typename T>
    class Convolutional : public Layer<T>
    {
    protected:
        const internal::ConvDims m_dim;
        Vector<T> m_filterData;
        Vector<Scalar> m_dfData;
        Vector<T> m_bias;
        Vector<Scalar> m_db;
        
        // Matrix m_out;
        // Matrix m_din;
    public:
        Convolutional(const int inWidth,const int inHeight,
            const int inChannels, const int outChannels, const int windowWidth, const int windowHeight):
        Layer<T>((inWidth * inHeight * inChannels), 
        (inWidth - windowWidth + 1) * (inHeight - windowHeight + 1) * outChannels), 
        m_dim(inChannels, outChannels, inHeight, inWidth, windowHeight, windowWidth)
        {
            Init();
        }
        ~Convolutional() {}

        virtual void Init() override
        {
            const int filterDataSize = m_dim.inChannels * m_dim.outChannels * m_dim.FilterRows * m_dim.FilterCols;

            m_filterData.resize(filterDataSize);
            m_dfData.resize(filterDataSize);

            m_bias.resize(m_dim.outChannels);
            m_db.resize(m_dim.outChannels);
        }

        void Init(const Scalar& mu, const Scalar& sigma, Random& rng) override
        {
            Init();
            const int filterDataSize = m_dim.inChannels * m_dim.outChannels * m_dim.FilterRows * m_dim.FilterCols;
            rng.SetNormalDistRandom(m_filterData,mu,sigma);
            rng.SetNormalDistRandom(m_bias,mu,sigma);
        }

        virtual Matrix<T> Forward(const Matrix<T>& InData) override
        {
            const int nObs = InData.cols();
            Matrix<T> m_out;
            m_out.resize(this->m_outSize, nObs);
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                internal::Convolve_Valid_Autodiff<T>(m_dim, InData, nObs, m_filterData, m_out);
            }
            else
            {       
                internal::Convolve_Valid(m_dim, InData.data(),true,nObs,m_filterData.data(),m_out.data());
            }
            int channelStartRow = 0;
            const int channelNElem = m_dim.ConvRows * m_dim.ConvCols;

            for (int i = 0; i < m_dim.outChannels; i++, channelStartRow += channelNElem)
            {
                m_out.block(channelStartRow, 0, channelNElem,nObs).array() += m_bias[i];
            }
            return m_out;
            
        }

        virtual void Backward(T& Loss) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                const int filterSize = m_filterData.size();
                const int biasSize = m_bias.size();
                const int totalSize = filterSize + biasSize;
                Eigen::Matrix<T, Eigen::Dynamic, 1> all_params(totalSize);
                all_params.head(filterSize) = m_filterData;
                all_params.tail(biasSize) = m_bias;

                // =================================================================
                //只调用一次 gradient()，让 autodiff 只遍历一遍计算图，性能最大化
                // =================================================================
                auto all_grads = autodiff::gradient(Loss, all_params);

                // =================================================================
                // 将计算出的一维总梯度向量，分别拆分回对应的 Scalar 梯度向量中
                // =================================================================
                for (int i = 0; i < filterSize; ++i)
                {
                    m_dfData[i] = all_grads[i];
                }

                for (int i = 0; i < biasSize; ++i)
                {
                    m_db[i] = all_grads[filterSize + i]; // 偏置的梯度接在权重后面
                }
            }
            // const int nObs = LastLayerData.cols();
            // z_j = sum_i(conv(in_i, w_ij)) + b_j
            //
            // d(z_k) / d(w_ij) = 0, if k != j
            // d(L) / d(w_ij) = [d(z_j) / d(w_ij)] * [d(L) / d(z_j)] = sum_i{ [d(z_j) / d(w_ij)] * [d(L) / d(z_j)] }
            // = sum_i(conv(in_i, d(L) / d(z_j)))
            //
            // z_j is an image (matrix), b_j is a scalar
            // d(z_j) / d(b_j) = a matrix of the same size of d(z_j) filled with 1
            // d(L) / d(b_j) = (d(L) / d(z_j)).sum()
            //
            // d(z_j) / d(in_i) = conv_full_op(w_ij_rotate)
            // d(L) / d(in_i) = sum_j((d(z_j) / d(in_i)) * (d(L) / d(z_j))) = sum_j(conv_full(d(L) / d(z_j), w_ij_rotate))
            /*internal::ConvDims backConvDim(
                nObs, m_dim.outChannels,m_dim.ChannelRows,
                m_dim.ChannelCols,m_dim.ConvRows, m_dim.ConvCols);
            internal::Convolve_Valid(backConvDim, LastLayerData.data(),false,m_dim.inChannels,BackPropData.data(),m_dfData.data());

            m_dfData /= nObs;
            // Derivative for bias
            // Aggregate d(L) / d(z) in each output channel
            ConstAlignedMapMat dOutByChannel(BackPropData.data(), m_dim.ConvRows * m_dim.ConvCols, m_dim.outChannels * nObs);
            Vector db = dOutByChannel.colwise().sum();

            //average
            ConstAlignedMapMat dbByObs(db.data(),m_dim.outChannels, nObs);
            m_db.noalias() = dbByObs.rowwise().mean();
            // Compute d(L) / d_in = conv_full(d(L) / d(z), w_rotate)
           m_din.resize(m_inSize, nObs);

            internal::ConvDims convFullDim(m_dim.outChannels, m_dim.inChannels, m_dim.ConvRows, m_dim.ConvCols,m_dim.FilterRows,m_dim.FilterCols);
            internal::Convolve_Full(convFullDim, BackPropData.data(), nObs, m_filterData.data(), m_din.data());*/
        }

        virtual void Update(Optimizer<Scalar>& opt) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                opt.Update(m_dfData,m_filterData);
                opt.Update(m_db,m_bias);
            }
        }

        virtual std::vector<Scalar> GetParameters() const override
        {
            std::vector<Scalar> res(m_filterData.size() + m_bias.size());

            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                Vector<Scalar> w(m_filterData.size());
                Vector<Scalar> b(m_bias.size());
                w = m_filterData.reshaped().unaryExpr([](const AutoDiffVar& x){ return x.expr->val; });
                b = m_bias.reshaped().unaryExpr([](const AutoDiffVar& x){ return x.expr->val; });
                std::vector<Scalar> params(m_filterData.size()+m_bias.size());
                std::copy(w.data(), w.data()+w.size(), params.begin());
                std::copy(b.data(), b.data()+b.size(), params.begin()+w.size());
                return params;
            }
            else
            {
                std::copy(m_filterData.data(), m_filterData.data() + static_cast<int>(m_filterData.size()), res.begin());
                std::copy(m_bias.data(), m_bias.data() + static_cast<int>(m_bias.size()), res.begin() + m_filterData.size());
                return res;
            }
        }

        virtual void SetParameters(const std::vector<Scalar>& param) override
        {
            if (static_cast<int>(param.size()) != m_filterData.size() + m_bias.size())
            {
                throw std::invalid_argument("[convolution]: parameter size mismatch");
            }
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                Vector<Scalar> w(m_filterData.size());
                Vector<Scalar> b(m_bias.size());
                std::copy(param.begin(), param.begin() + static_cast<int>(m_filterData.size()), w.data());
                std::copy(param.begin() + static_cast<int>(m_filterData.size()), param.end(), b.data());
                m_filterData = w.unaryExpr([](const Scalar& x){ return AutoDiffVar(x); });
                m_bias = b.unaryExpr([](const Scalar& x){ return AutoDiffVar(x); });
            }
            else
            {
                std::copy(param.begin(),param.begin()+static_cast<int>(m_filterData.size()),m_filterData.data());
                std::copy(param.begin()+static_cast<int>(m_filterData.size()), param.end(),m_bias.data());
            }
        }

        virtual std::string GetSubType()const override{return "Convolution";}
    };
} // namespace MiniBrain
