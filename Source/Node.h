#pragma once
#include "Eigen/Dense"

namespace MiniBrain 
{
    
    class Node 
    {
        public:
            Node(){}

            virtual ~Node(){}
            
            virtual std::string GetType()const {return "Node";}
            virtual std::string GetSubType()const{return "";}
    };    
}