#include <iostream>
#include "MiniBrain.h"
#include "Utils/IO.h"

using namespace MiniBrain;

int main(int argc, char const *argv[])
{
    Network<AutoDiffVar> nn,nn2;
    Layer<AutoDiffVar>* layer1 = new FullyConnected<AutoDiffVar>(1,6);
    Layer<AutoDiffVar>* layer2 = new FullyConnected<AutoDiffVar>(6,1);
    // GRU* layer3 = new GRU(1,6);
    // Layer* layer4 = new FullyConnected(12,12);
    // Convolutional<AutoDiffVar>* layer5 = new Convolutional<AutoDiffVar>(10,10,1,2,3,3);
    // Layer<AutoDiffVar>* layer6 = new FullyConnected<AutoDiffVar>(8*8*2,2);

    Activation<AutoDiffVar>* a1 = new ReLU<AutoDiffVar>();
    
    nn.AddLayer(layer1);
    nn.AddLayer(a1);
    // nn.AddLayer(layer4);
    // nn.AddLayer(layer3);
    nn.AddLayer(layer2);

    // Layer* layer7 = new Convolutional(10,10,1,2,3,3);
    // Layer* layer8 = new FullyConnected(8*8*2,2);
    // nn2.AddLayer(layer7);
    // nn2.AddLayer(layer8);

    nn.SetLossFunc(new RegressionMSE());

    Adam opt;

    nn.Init(0,0.01);

    // std::string path = "test.bin";

    // std::cout<<"before:\n";
    // for (std::vector<Scalar> param : nn.GetParameters())
    // {        
    //     for (Scalar p : param)
    //     {
    //         std::cout<<p<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<"========"<<std::endl;
    // for (std::vector<Scalar> param : nn2.GetParameters())
    // {        
    //     for (Scalar p : param)
    //     {
    //         std::cout<<p<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    // io::SaveParameter(&nn,path);

    // io::LoadParameter(&nn2,path);

    // std::cout<<"after:\n";
    // for (std::vector<Scalar> param : nn2.GetParameters())
    // {        
    //     for (Scalar p : param)
    //     {
    //         std::cout<<p<<" ";
    //     }
    //     std::cout<<std::endl;
    // }


    // Matrix<AutoDiffVar> x = Matrix<AutoDiffVar>::Random(2,2);
    // Matrix<AutoDiffVar> y = Matrix<AutoDiffVar>::Random(1,2);
    // Matrix x2 = Matrix::Random(100,2);
    Matrix<AutoDiffVar> x(1,6);
    Matrix<AutoDiffVar> y(1,6);

    x<<1,   2,
        3,4,
       5,6;
    y<<2, 3,
        4, 5,6,7;

    // std::cout<<x<<std::endl;

    // layer3->SetBatchSize(1);
    // // Matrix in = Matrix::Random(2,1);
    Matrix<AutoDiffVar> output = nn.Forward(x);
    std::cout<<y<<std::endl;
    // // layer3->SetBatchSize(1);
    for (int i = 0; i < 10; i++)
    {
        Matrix<AutoDiffVar> roundOut = nn.Forward(x);
        nn.Backward(roundOut, y);
        nn.Update(opt);
    }   
    
    // Matrix in(1,6);
    // in<<5,6,7,8,9,10;

    std::cout<<"before:\n";
    std::cout<< output<<std::endl<<"after:\n"<<nn.Forward(x)<<std::endl;
    
    return 0;
}
