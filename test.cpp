#include <iostream>
#include "Source/MiniBrain.h"
using namespace MiniBrain;

int main(int argc, char const *argv[])
{
    Network nn;
    Layer* layer1 = new FullyConnected(3,5);
    Layer* layer2 = new FullyConnected(5,1);

    Activation* a1 = new ReLU();
    
    nn.AddLayer(layer1);
    nn.AddLayer(a1);
    nn.AddLayer(layer2);

    nn.SetLossFunc(new RegressionMSE());

    Adam opt;

    
    return 0;
}
