# MiniBrain

MiniBrain 是一个 header-only 的神经网络库，基于 Eigen 和 autodiff 自动微分。

## 特性

- Header-only：仅通过头文件即可使用，无需额外库链接
- 基于 Eigen：使用 Eigen 矩阵库进行线性代数运算
- 支持 autodiff 自动微分
- 支持以下网络层类型：
  - 全连接层（Fully Connected）
  - 单层 GRU
  - 单头注意力（Single-Head Attention）
  - 卷积层（Convolutional）

## 依赖

- Eigen 3.4.1（已包含于 `Source/Eigen`）
- C++17 编译器

## 编译

使用支持 C++17 的编译器编译测试或示例代码，并包含头文件目录：

```bash
clang++ -std=c++17 -I Source -I Source/ThirdParty test.cpp -o test.exe
```

## 简单使用示例

示例：

```cpp
#include <iostream>
#include "MiniBrain.h"

using namespace MiniBrain;

int main()
{
    Network<AutoDiffVar> nn;

    nn.AddLayer(std::make_unique<FullyConnected<AutoDiffVar>>(1, 6));
    nn.AddLayer(std::make_unique<ReLU<AutoDiffVar>>());
    nn.AddLayer(std::make_unique<FullyConnected<AutoDiffVar>>(6, 1));

    nn.SetLossFunc(std::make_unique<RegressionMSE>());
    Adam opt;
    nn.Init(0, 0.01);

    Matrix<AutoDiffVar> x(1, 6);
    Matrix<AutoDiffVar> y(1, 6);
    x << 1, 2, 3, 4, 5, 6;
    y << 2, 3, 4, 5, 6, 7;

    auto before = nn.Forward(x);

    for (int i = 0; i < 600; ++i) {
        auto output = nn.Forward(x);
        nn.Backward(output, y);
        nn.Update(opt);
    }

    auto after = nn.Forward(x);

    std::cout << "before:\n" << before << "\n";
    std::cout << "after:\n" << after << "\n";
    return 0;
}
```

## 使用方式

将项目根目录加入包含路径后即可包含头文件并直接使用。