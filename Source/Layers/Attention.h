#ifndef ATTENTION_H
#define ATTENTION_H
#include <autodiff/reverse/var/eigen.hpp>
#include "../Layer.h"
#include "../Optimizer.h"
#include <memory>
#include <vector>

namespace MiniBrain
{
    template<typename T>
    class Attention : public Layer<T>
    {
        int m_seqLen;
        int m_featureCount;
        // 权重矩阵
        Matrix<T> Wq, Wk, Wv;
        Matrix<Scalar> dWq, dWk, dWv;
    public:
        Attention(int inputSize, int outputSize, int featureCount):Layer<T>(inputSize, outputSize)
        {
            m_seqLen = inputSize / featureCount;
            m_featureCount = featureCount;
            Init();
        }
        
        virtual void Init() override
        {
            Wq.resize(m_featureCount, this->m_inSize);
            Wk.resize(m_featureCount, this->m_inSize);
            Wv.resize(this->m_outSize,this->m_inSize);

            dWk.resize(m_featureCount, this->m_inSize);
            dWq.resize(m_featureCount,this->m_inSize);
            dWv.resize(this->m_outSize, this->m_inSize);
        }

        virtual void Init(const Scalar& mu, const Scalar& sigma, Random& RNG) override
        {
            Init();
            RNG.SetNormalDistRandom(Wq, mu, sigma);
            RNG.SetNormalDistRandom(Wk, mu, sigma);
            RNG.SetNormalDistRandom(Wv, mu, sigma);
        }

        virtual Matrix<T> Forward(const Matrix<T>& InData) override
        {
            Matrix<T> output;
            const int batchSize  = InData.cols;
            output.resize(this->m_outSize*m_seqLen, batchSize);
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                // 缩放因子：1 / sqrt(d_k)
                T scale = 1.0 / std::sqrt(static_cast<Scalar>(this->m_outputSize));

                // 按样本(Batch)进行分块处理，内部完全基于原生 Eigen 矩阵乘法
                for (int o = 0; o < batchSize; ++o)
                {
                    // 1. 从拼合输入中恢复当前样本的序列矩阵 X_obs [m_featureCount, m_seqLen]
                    // 这里必须用一条链式赋值，让 autodiff 跟踪到这个样本的计算图节点
                    Matrix<T> X_obs(m_featureCount, m_seqLen);
                    for (int s = 0; s < m_seqLen; ++s) {
                        X_obs.col(s) = InData.block(s * m_featureCount, o, m_featureCount, 1);
                    }

                    // 2. 线性投影得到 Q, K, V 矩阵
                    // Q, K, V 的形状均为 [m_outputSize, m_seqLen]
                    Matrix<T> Q = Wq * X_obs;
                    Matrix<T> K = Wk * X_obs;
                    Matrix<T> V = Wv * X_obs;

                    // 3. 计算注意力得分矩阵 Score = Q^T * K, 形状为 [m_seqLen, m_seqLen]
                    // 这一步是标准的矩阵乘法，autodiff 会完美捕获并生成一个整体大节点
                    Matrix<T> Score = Q.transpose() * K;
                    Score.array() *= scale; // 缩放

                    // 4. 对 Score 的每一列进行标准的 Softmax 操作（矩阵化无循环/低循环实现）
                    // 为了数值稳定性，减去每列的最大值
                    for (int s = 0; s < m_seqLen; ++s) {
                        T max_val = Score.col(s).maxCoeff();
                        // 逐元素求指数（对 autodiff::var / varf 极其友好，会自动注册 exp 导数）
                        Score.col(s) = (Score.col(s).array() - max_val).exp();
                        T sum_val = Score.col(s).sum();
                        Score.col(s).array() /= sum_val;
                    }
                    // 此时 Score 变为了 Attention Weights (A)

                    // 5. 加权聚合：Context = V * A^T, 形状为 [m_outputSize, m_seqLen]
                    // 注意：原标准公式是 A * V，但因为我们这里维度排布习惯，
                    // V 是 [d_v, SeqLen]，A 是 [SeqLen, SeqLen]，所以是 V * A.transpose()
                    Matrix<T> Context = V * Score.transpose();

                    // 6. 将当前样本的结果拼装回到输出矩阵 output 对应的列中
                    for (int s = 0; s < m_seqLen; ++s) {
                        output.block(s * this->m_outputSize, o, this->m_outputSize, 1) = Context.col(s);
                    }
                }
            }
            else
            {
                const Scalar scale =  1.0f / std::sqrt(static_cast<Scalar>(this->m_outputSize));

                for (int b = 0; b < batchSize; ++b)
                {
                    // 拆包
                    Eigen::Map<const Matrix<T>> x(InData.col(b).data(), m_featureCount, m_seqLen);
                    // Q K V
                    Matrix<T> q = Wq * x;      // [d_k, seq_len]
                    Matrix<T> k = Wk * x;      // [d_k, seq_len]
                    Matrix<T> v = Wv * x;      // [output_size, seq_len]

                    // Attention Score
                    Matrix<T> score =  q.transpose() * k;
                    score *= scale;

                    // Softmax
                    for (int r = 0; r < m_seqLen; ++r)
                    {
                        Scalar maxv = score.row(r).maxCoeff();
                        auto exps = (score.row(r).array() - maxv).exp();
                        score.row(r) = exps / exps.sum();
                    }

                    // 输出
                    Matrix<T> y = v * score.transpose();

                    // 重新拼包
                    Eigen::Map<Vector<T>>(output.col(b).data(),this->m_outputSize * m_seqLen) = Eigen::Map<Vector<T>>(y.data(), this->m_outputSize * m_seqLen);
                }}


            return output;
        }

        void Backward(T& Loss) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                const int wqSize = Wq.size();
                const int wkSize = Wk.size();
                const int wvSize = Wv.size();
                const int totalSize = wqSize + wkSize + wvSize;

                // =================================================================
                // 1. 高性能拼合：将三组权重矩阵的 var/varf 节点融合成一个大一维向量
                // =================================================================
                Eigen::Matrix<T, Eigen::Dynamic, 1> all_params(totalSize);
                
                // 使用 Map 或 block 将二维矩阵无缝展平存入大向量（保留计算图指针）
                // Eigen 的 Matrix 默认是列优先存储，直接用 reshape 形式或者 block 即可
                all_params.segment(0, wqSize) = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(Wq.data(), wqSize);
                all_params.segment(wqSize, wkSize) = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(Wk.data(), wkSize);
                all_params.segment(wqSize + wkSize, wvSize) = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(Wv.data(), wvSize);

                // =================================================================
                // 2. 核心调用：单次触发 gradient() 算法，一次遍历扫完整个复杂的 Attention 计算图
                // =================================================================
                auto all_grads = autodiff::gradient(Loss, all_params);

                // =================================================================
                // 3. 分流拆分：将 Scalar 纯数值拷贝并对齐到给优化器用的梯度向量中
                // =================================================================
                // 提取 Wq 的梯度
                for (int i = 0; i < wqSize; ++i) {
                    dWq[i] = all_grads[i];
                }
                
                // 提取 Wk 的梯度
                for (int i = 0; i < wkSize; ++i) {
                    dWk[i] = all_grads[wqSize + i];
                }

                // 提取 Wv 的梯度
                for (int i = 0; i < wvSize; ++i) {
                    dWv[i] = all_grads[wqSize + wkSize + i];
                }
            }
        }

        virtual void Update(Optimizer<Scalar>& opt) override
        {
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                opt.Update(dWq, Wq);
                opt.Update(dWk, Wk);
                opt.Update(dWv, Wv);
            }
        }

        virtual std::vector<Scalar> GetParameters() const override
        {
            const int wqSize = Wq.size();
            const int wkSize = Wk.size();
            const int wvSize = Wv.size();
            const int totalSize = wqSize + wkSize + wvSize;

            std::vector<Scalar> params(totalSize);

            // 只要模板 T 是计算图变量类型（对齐你的别名 AutoDiffVar，也可以写 !std::is_floating_point_v<T>）
            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                // 1. 提取 QKV 矩阵内部的纯数值（Scalar）
                Vector<Scalar> wq = Wq.reshaped().unaryExpr([](const AutoDiffVar& x){ return x.expr->val; });
                Vector<Scalar> wk = Wk.reshaped().unaryExpr([](const AutoDiffVar& x){ return x.expr->val; });
                Vector<Scalar> wv = Wv.reshaped().unaryExpr([](const AutoDiffVar& x){ return x.expr->val; });

                // 2. 依次拷贝到统一的序列化 vector 中
                std::copy(wq.data(), wq.data() + wqSize, params.begin());
                std::copy(wk.data(), wk.data() + wkSize, params.begin() + wqSize);
                std::copy(wv.data(), wv.data() + wvSize, params.begin() + wqSize + wkSize);
            }
            else
            {
                // 纯推理模式 (T = float/double)：直接通过裸指针 data() 物理内存拷贝
                std::copy(Wq.data(), Wq.data() + wqSize, params.begin());
                std::copy(Wk.data(), Wk.data() + wkSize, params.begin() + wqSize);
                std::copy(Wv.data(), Wv.data() + wvSize, params.begin() + wqSize + wkSize);
            }

            return params;
        }


        virtual void SetParameters(const std::vector<Scalar>& param) override
        {
            const int wqSize = Wq.size();
            const int wkSize = Wk.size();
            const int wvSize = Wv.size();
            const int totalSize = wqSize + wkSize + wvSize;

            // 1. 严格尺寸校验
            if (static_cast<int>(param.size()) != totalSize)
            {
                throw std::invalid_argument("AttentionLayer: parameter size mismatch");
            }

            if constexpr (std::is_same_v<T, AutoDiffVar>)
            {
                // 2. 训练模式：使用 .reshaped()(i) 的左值按元素赋值，这会安全地覆盖 varf/var 的内部数值而不会破坏矩阵本身结构
                for (int i = 0; i < wqSize; i++)
                {
                    Wq.reshaped()(i) = param[i];
                }
                for (int i = 0; i < wkSize; i++)
                {
                    Wk.reshaped()(i) = param[wqSize + i];
                }
                for (int i = 0; i < wvSize; i++)
                {
                    Wv.reshaped()(i) = param[wqSize + wkSize + i];
                }
            }
            else
            {
                // 3. 推理模式：使用 std::copy 直接物理覆盖 float 内存，速度极快
                std::copy(param.begin(), param.begin() + wqSize, Wq.data());
                std::copy(param.begin() + wqSize, param.begin() + wqSize + wkSize, Wk.data());
                std::copy(param.begin() + wqSize + wkSize, param.end(), Wv.data());
            }
        }


        virtual std::string GetSubType()const override{return "Attention";} 
    };
}

#endif // ATTENTION_H