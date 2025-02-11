#pragma once
#include "../Eigen/Dense"
#include "../Optimizer.h"
#include <map>

namespace MiniBrain
{
    class Adam : public Optimizer
    {
    protected:
        std::map<const float*,Array> m_history_m;
        std::map<const float*,Array> m_history_v;
        float m_beta_1t;
        float m_beta_2t;

    public:
        float m_learnRate;
        float m_eps;
        float m_beta1;
        float m_beta2;

        Adam(const float& lrate = 0.001f, const float& eps = float(1e-6),
            const float& beta1 = 0.9f, const float& beta2 = 0.999f):
            m_beta_1t(beta1), m_beta_2t(beta2),
            m_learnRate(lrate), m_eps(eps),
            m_beta1(beta1), m_beta2(beta2) 
        {}

        ~Adam() {}

        virtual void Reset() override
        {
            m_history_m.clear();
            m_history_v.clear();
            m_beta_1t = m_beta1;
            m_beta_2t = m_beta2;
        }

        virtual void Update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) override
        {
            // Get the m and v vectors associated with this gradient
            Array& mvec = m_history_m[dvec.data()];
            Array& vvec = m_history_v[dvec.data()];
            //init if len = 0
            if (mvec.size() == 0)
            {
                mvec.resize(dvec.size());
                mvec.setZero();
            }

            if (vvec.size() == 0)
            {
                vvec.resize(dvec.size());
                vvec.setZero();
            }
            
            // update m and v
            mvec = m_beta1*mvec+(1.0f-m_beta1)*dvec.array();
            vvec = m_beta2*vvec+(1.0f-m_beta2)*dvec.array().square();
            // Correction coeffients
            const float correct1 = 1.0f / (1.0f - m_beta_1t);
            const float correct2 = 1.0f / std::sqrt(1.0f - m_beta_2t);
            // update parameters
            vec.array() -= (m_learnRate * correct1) * mvec / (correct2 * vvec.sqrt() + m_eps);
            m_beta_1t *= m_beta1;
            m_beta_2t *- m_beta2;
        }
    };
} // namespace MiniBrain
