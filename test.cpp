#include <iostream>
#include "Source/MiniBrain.h"
using namespace MiniBrain;

int main(int argc, char const *argv[])
{
    Network nn;
    Layer* layer1 = new FullyConnected(2,2);
    Layer* layer2 = new FullyConnected(6,1);
    GRU* layer3 = new GRU(1,6);
    Layer* layer4 = new FullyConnected(12,12);

    Activation* a1 = new ReLU();
    
    // nn.AddLayer(layer1);
    // nn.AddLayer(a1);
    // nn.AddLayer(layer4);
    nn.AddLayer(layer3);
    nn.AddLayer(layer2);

    nn.SetLossFunc(new RegressionMSE());

    Adam opt;

    nn.Init(0,0.01);

    layer3->SetBatchSize(6);
    layer3->ResetMemory();

    // Matrix x = Matrix::Random(1,200);
    // Matrix y = Matrix::Random(2,1);

    Matrix x(1,6);
    Matrix y(1,6);

    x<<1,   2,
        3,4,
       5,6;
    y<<2, 3,
        4, 5,6,7;

    // std::cout<<x<<std::endl;

    // layer3->SetBatchSize(1);
    // Matrix in = Matrix::Random(2,1);
    Matrix output = nn.Predict(x);
    // layer3->SetBatchSize(1);
    for (int i = 0; i < 300; i++)
    {
        nn.Predict(x);
        nn.Backward(x,y);
        nn.Update(opt);
    }   
    
    Matrix in(1,6);
    in<<5,6,7,8,9,10;

    std::cout<<"before:\n";
    std::cout<< output<<std::endl<<"after:\n"<<nn.Predict(in)<<std::endl;
    
    return 0;
}
