#include <iostream>
#include "Eigen/Dense"

int main(int argc, char const *argv[])
{
    Eigen::MatrixXf m(2,2);
    m<<1,2,
    3,4;

    std::cout<<"m: "<<m<<std::endl;
    return 0;
}
