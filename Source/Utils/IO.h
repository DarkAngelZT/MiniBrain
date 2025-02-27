#pragma once
#include "../Network.h"
#include "bitsery/bitsery.h"
#include "bitsery/adapter/stream.h"
#include <fstream>

namespace MiniBrain
{
    namespace io
    {
        class NetworkParamData
        {
            std::vector<float> data;
            friend class bitsery::Access;
            NetworkParamData()=default;

            template<typename S>
            void serialize(S& s)
            {

            }
        public:
            NetworkParamData(){}
        };

        void SaveParameter(Network* network,std::string path)
        {

        }

        void ReadParameter(Network* network,std::string path)
        {

        }
    } // namespace internal
    
} // namespace MiniBrain
