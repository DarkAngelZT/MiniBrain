#include "../Eigen/Dense"
#include "../Layer.h"

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

        //合并格式，z=w*x+b
        Matrix m_z;
        //当前层的输出
        Matrix m_out;
        //输入端的导数
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
    };
}