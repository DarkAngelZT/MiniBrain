#pragma once
#include <random>

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

        void SetNormalDistRandom(
            float* OutArray, const int size,
            const float& mu = 0.f, const float& sigma = 1.f)
        {
            std::normal_distribution<float> Dist(mu, sigma);
            for (int i = 0; i < size; i++)
            {
                OutArray[i] = Dist(m_gen);
            }            
        }

        float Rand()
        {
            std::uniform_real_distribution<float> Dist{0.0f,1.0f};
            return Dist(m_gen);
        }

        int RandInt(int Min, int Max)
        {
            std::uniform_int_distribution<> Dist(Min,Max);
            return Dist(m_gen);
        }
    };
} // namespace MiniBrain
