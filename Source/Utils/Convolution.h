#pragma once
#include "../TypeDef.h"


namespace MiniBrain
{
    namespace internal
    {
        struct ConvDims
        {
            const int inChannels;
            const int outChannels;
            const int ChannelRows;
            const int ChannelCols;
            const int FilterRows;
            const int FilterCols;

            const int ImgRows;
            const int ImgCols;

            const int ConvRows;
            const int ConvCols;

            ConvDims(const int in_channels,const int out_channels, const int channel_rows, const int channel_cols,
            const int filter_rows, const int filter_cols):
            inChannels(in_channels),outChannels(out_channels),ChannelRows(channel_rows),ChannelCols(channel_cols),
            FilterRows(filter_rows),FilterCols(filter_cols),
            ImgRows(channel_rows),ImgCols(in_channels * channel_cols),
            ConvRows(channel_rows - filter_rows + 1), ConvCols(channel_cols - filter_cols + 1)
            {}
        };
        
        inline void FlattenMat(const ConvDims& Dim, const float* Src, const int stride, const int nObs,
            Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& FlatMat)
        {
            const int& SegmentSize = Dim.FilterRows;
            const std::size_t copyBytes = sizeof(float)*SegmentSize;
            float* Writer = FlatMat.data();
            const int ChannelSize = Dim.ChannelRows * Dim.ChannelCols;

            for (int i = 0; i < nObs; i++)
            {
                const float* ReaderRow = Src;
                const float* const ReaderRowEnd = Src+Dim.ConvRows;
                for ( ; ReaderRow < ReaderRowEnd; ReaderRow++)
                {
                    const float* Reader = ReaderRow;
                    const float* const ReaderEnd = Reader + ChannelSize;

                    for (; Reader < ReaderEnd; Reader += Dim.ChannelRows, Writer += SegmentSize)
                    {
                        std::memcpy(Writer, Reader, copyBytes);
                    }
                    
                }
                
            }
            
        }
    } // namespace internal
    
} // namespace MiniBrain
