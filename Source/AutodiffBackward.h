#pragma once

#include <autodiff/reverse/var/eigen.hpp>

#include "Network.h"

namespace MiniBrain
{
    /// Compute one reverse pass for every supplied network.
    ///
    /// Layer::Backward historically calls autodiff::gradient once per layer.
    /// That repeats the complete loss-graph traversal for GRU, Attention and
    /// every FullyConnected layer. This helper aliases all parameter ExprPtr
    /// objects into one vector, performs one reverse pass, and then routes each
    /// contiguous gradient slice back to its owning layer.
    ///
    /// No GRU/Attention derivative is implemented here: autodiff still derives
    /// every gradient from the forward graph. The networks only agree on a
    /// stable parameter order for packing and unpacking.
    template<typename... Networks>
    void BackwardOnce(AutoDiffVar& loss, Networks&... networks)
    {
        static_assert(
            (std::is_same_v<Networks, Network<AutoDiffVar>> && ...),
            "BackwardOnce accepts Network<AutoDiffVar> objects only");

        const int total_parameters =
            (0 + ... + networks.GetAutoDiffParameterCount());
        if(total_parameters <= 0)
            return;

        Vector<AutoDiffVar> parameters(total_parameters);
        int offset = 0;
        (networks.AppendAutoDiffParameters(parameters, offset), ...);
        if(offset != total_parameters)
            MINIBRAIN_THROW(std::logic_error("BackwardOnce: parameter append offset mismatch"));

        const Vector<Scalar> gradients = autodiff::gradient(loss, parameters);
        if(gradients.size() != total_parameters)
            MINIBRAIN_THROW(std::logic_error("BackwardOnce: gradient size mismatch"));

        offset = 0;
        (networks.AssignGradients(gradients, offset), ...);
        if(offset != total_parameters)
            MINIBRAIN_THROW(std::logic_error("BackwardOnce: gradient assign offset mismatch"));
    }
}

