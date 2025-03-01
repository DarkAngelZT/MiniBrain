#pragma once
#include "../Network.h"
#include "bitsery/bitsery.h"
#include "bitsery/adapter/stream.h"
#include "bitsery/traits/vector.h"
#include <fstream>

namespace MiniBrain
{
    #define MAX_PARAM_SIZE INT_MAX
    #define MAX_LAYER_SIZE 1000
    namespace io
    {
        class SingleParamData
        {
            public:
            std::vector<float> data;
            friend class bitsery::Access;

            template<typename S>
            void serialize(S& s)
            {
                s.container4b(data,MAX_PARAM_SIZE);
            }
        
            SingleParamData(){}
        };

        class NetworkData
        {
            public:
            std::vector<SingleParamData> data;
            friend class bitsery::Access;

            template<typename S>
            void serialize(S& s)
            {
                s.container(data,MAX_LAYER_SIZE,[](S& s,SingleParamData& d)
                {
                    s.object(d);
                });
            }
            
            NetworkData() {}
        };

        void SaveParameter(Network* network,std::string path)
        {
            std::vector<std::vector<float>> data = network->GetParameters();
            std::ofstream file(path,std::ios::binary);
            bitsery::Serializer<bitsery::OutputBufferedStreamAdapter> ser(file);
            NetworkData nd;
            for(auto& d:data)
            {
                SingleParamData spd;
                spd.data = d;
                nd.data.push_back(spd);
            }
            ser.object(nd);
            ser.adapter().flush();
            file.close();
        }

        void LoadParameter(Network* network,std::string path)
        {
            std::ifstream file(path,std::ios::binary);
            NetworkData nd;
            bitsery::Deserializer<bitsery::InputStreamAdapter> des(file);
            des.object(nd);
            file.close();
            if (nd.data.size() == 0)
            {
                return;
            }
            
            std::vector<std::vector<float>> data;
            for(auto& d:nd.data)
            {
                data.push_back(d.data);
            }
            network->SetParameters(data);
        }
    } // namespace internal
    
} // namespace MiniBrain
