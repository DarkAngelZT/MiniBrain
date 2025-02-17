#include <iostream>
#include "Source/MiniBrain.h"
using namespace MiniBrain;

int main(int argc, char const *argv[])
{
    Network nn;
    Layer* layer1 = new FullyConnected(4,6);
    Layer* layer2 = new FullyConnected(6,2);

    Activation* a1 = new ReLU();
    
    nn.AddLayer(layer1);
    nn.AddLayer(a1);
    nn.AddLayer(layer2);

    nn.SetLossFunc(new RegressionMSE());

    Adam opt;

    nn.Init(0,0.01);

    Matrix x = Matrix::Random(4,2);

    Matrix output = nn.Predict(x);

    std::cout<< output<<std::endl;
    
    return 0;
}
