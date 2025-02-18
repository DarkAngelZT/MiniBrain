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
    Matrix y = Matrix::Random(2,2);

    Matrix output = nn.Predict(x);

    nn.Backward(x,y);
    nn.Update(opt);

    Matrix output2 = nn.Predict(x);

    std::cout<<"target:\n"<<y<<std::endl<<"before:\n";
    std::cout<< output<<std::endl<<"after:\n"<<output2<<std::endl;
    
    return 0;
}
