#pragma once
#include "../Eigen/Dense"
#include "../Layer.h"
#include "../Utils/Convolution.h"
#include "../Optimizer.h"

namespace MiniBrain
{
    class Convolutional : public Layer
    {
    protected:
        const internal::ConvDims m_dim;
        Vector m_filterData;
        Vector m_dfData;
        Vector m_bias;
        Vector m_db;
        
        Matrix m_out;
        Matrix m_din;
    public:
        Convolutional(const int inWidth,const int inHeight,
            const int inChannels, const int outChannels, const int windowWidth, const int windowHeight):
        Layer(inWidth * inHeight * inChannels, 
        (inWidth - windowWidth + 1) * (inHeight - windowHeight + 1) * outChannels), 
        m_dim(inChannels, outChannels, inHeight, inWidth, windowHeight, windowWidth)
        {
            Init();
        }
        ~Convolutional() {}

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
            const int filterDataSize = m_dim.inChannels * m_dim.outChannels * m_dim.FilterRows * m_dim.FilterCols;

            m_filterData.resize(filterDataSize);
            m_dfData.resize(filterDataSize);

            m_bias.resize(m_dim.outChannels);
            m_db.resize(m_dim.outChannels);
        }

        void Init(const float& mu, const float& sigma, Random& rng) override
        {
            Init();
            const int filterDataSize = m_dim.inChannels * m_dim.outChannels * m_dim.FilterRows * m_dim.FilterCols;
            rng.SetNormalDistRandom(m_filterData.data(),filterDataSize,mu,sigma);
            rng.SetNormalDistRandom(m_bias.data(), m_dim.outChannels,mu,sigma);
        }

        virtual void Forward(const Matrix& InData) override
        {
            const int nObs = InData.cols();

            m_out.resize(m_outSize, nObs);

            internal::Convolve_Valid(m_dim, InData.data(),true,nObs,m_filterData.data(),m_out.data());
            int channelStartRow = 0;
            const int channelNElem = m_dim.ConvRows * m_dim.ConvCols;

            for (int i = 0; i < m_dim.outChannels; i++, channelStartRow += channelNElem)
            {
                m_out.block(channelStartRow, 0, channelNElem,nObs).array() += m_bias[i];
            }
        }

        virtual void Backward(const Matrix& LastLayerData,const Matrix& BackPropData) override
        {
            const int nObs = LastLayerData.cols();
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
            internal::ConvDims backConvDim(
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
            internal::Convolve_Full(convFullDim, BackPropData.data(), nObs, m_filterData.data(), m_din.data());
        }

        virtual void Update(Optimizer&opt) override
        {
            ConstAlignedMapVec dw(m_dfData.data(),m_dfData.size());
            ConstAlignedMapVec db(m_db.data(),m_db.size());
            AlignedMapVec w(m_filterData.data(),m_filterData.size());
            AlignedMapVec b(m_bias.data(),m_bias.size());
            opt.Update(dw,w);
            opt.Update(db,b);
        }

        virtual std::vector<float> GetParameters() const override
        {
            std::vector<float> res(m_filterData.size() + m_bias.size());

            std::copy(m_filterData.data(), m_filterData.data() + static_cast<int>(m_filterData.size()), res.begin());
            std::copy(m_bias.data(), m_bias.data() + static_cast<int>(m_bias.size()), res.begin() + m_filterData.size());
            return res;
        }

        virtual void SetParameters(const std::vector<float>& param) override
        {
            if (static_cast<int>(param.size()) != m_filterData.size() + m_bias.size())
            {
                throw std::invalid_argument("[convolution]: parameter size mismatch");
            }
            std::copy(param.begin(),param.begin()+static_cast<int>(m_filterData.size()),m_filterData.data());
            std::copy(param.begin()+static_cast<int>(m_filterData.size()), param.end(),m_bias.end());
        }

        virtual std::string GetSubType()const override{return "Convolution";}
    };
} // namespace MiniBrain
