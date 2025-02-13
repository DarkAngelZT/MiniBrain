#pragma once

#include "Eigen/Core"

#include "Node.h"

#include "Layer.h"
#include "Layers/FullyConnected.h"

#include "Activation.h"
#include "Activations/Mish.h"
#include "Activations/ReLU.h"
#include "Activations/Tanh.h"

#include "Optimizer.h"
#include "Optimizer/Adam.h"

#include "LossFunc.h"
#include "LossFunc/RegressionMSE.h"
#include "LossFunc/CrossEntropy.h"

#include "Network.h"
