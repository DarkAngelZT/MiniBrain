#pragma once
#include <random>
#include "TypeDef.h"

namespace MiniBrain
{
    class Random
    {
    protected:
        std::random_device m_rd{};
        std::mt19937 m_gen;
    public:
        Random() 
        {
            m_gen = std::mt19937(m_rd());
        }
        ~Random() {}

        template<typename T>
        void SetNormalDistRandom(
            Eigen::MatrixBase<T>& OutArray,
            const Scalar& mu = 0.f, const Scalar& sigma = 1.f)
        {
            std::normal_distribution<Scalar> Dist(mu, sigma);

             if constexpr (std::is_same_v<T, AutoDiffVar>)
             {
                 for (int i = 0; i < OutArray.size(); i++)
                 {
                    OutArray(i) = static_cast<Scalar>(Dist(m_gen));
                 }
             }
             else
             {
                 for (int i = 0; i < OutArray.size(); i++)
                 {
                    OutArray(i) = Dist(m_gen);
                 }
             }
        }

        Scalar Rand()
        {
            std::uniform_real_distribution<Scalar> Dist{0.0f,1.0f};
            return Dist(m_gen);
        }

        int RandInt(int Min, int Max)
        {
            std::uniform_int_distribution<> Dist(Min,Max);
            return Dist(m_gen);
        }
    };
} // namespace MiniBrain
