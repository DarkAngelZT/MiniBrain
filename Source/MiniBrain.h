#pragma once

#include "Eigen/Core"

#include "TypeDef.h"
#include "Node.h"
#include "ComputeNode.h"

#include "Layer.h"
#include "Layers/FullyConnected.h"

#include "Activation.h"
#include "Activations/Mish.h"
#include "Activations/ReLU.h"
#include "Activations/Tanh.h"
#include "Activations/Sigmoid.h"

#include "Optimizer.h"
#include "Optimizer/Adam.h"

#include "LossFunc.h"
#include "LossFunc/RegressionMSE.h"
#include "LossFunc/CrossEntropy.h"

#include "Network.h"
