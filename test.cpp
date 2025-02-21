#include <iostream>
#include "Source/MiniBrain.h"
using namespace MiniBrain;

int main(int argc, char const *argv[])
{
    Network nn;
    Layer* layer1 = new FullyConnected(2,2);
    Layer* layer2 = new FullyConnected(12,2);
    GRU* layer3 = new GRU(2,2);

    Activation* a1 = new ReLU();
    
    nn.AddLayer(layer1);
    // nn.AddLayer(a1);
    // nn.AddLayer(layer3);
    // nn.AddLayer(layer2);

    nn.SetLossFunc(new RegressionMSE());

    Adam opt;

    nn.Init(0,0.01);

    layer3->SetBatchSize(1);
    layer3->ResetMemory();

    Matrix x = Matrix::Random(2,1);
    Matrix y = Matrix::Random(2,1);

    // Matrix x(4,2);
    // Matrix y(2,2);

    // x<<-0.997497,   0.170019,
    //     0.127171, -0.0402539,
    //     -0.613392,  -0.299417,
    //     0.617481,   0.791925;
    // y<<0.64568, -0.651784,
    //     0.49321,  0.717887;

    // std::cout<<x<<std::endl;

    Matrix output = nn.Predict(x);

    for (int i = 0; i < 270; i++)
    {
        nn.Predict(x);
        nn.Backward(x,y);
        nn.Update(opt);
    }   
    

    Matrix output2 = nn.Predict(x);

    std::cout<<"target:\n"<<y<<std::endl<<"before:\n";
    std::cout<< output<<std::endl<<"after:\n"<<output2<<std::endl;
    
    return 0;
}
