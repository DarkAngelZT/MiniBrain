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

        // A special matrix product. We select a window from 'mat1' and calculates its product with 'mat2',
        // and progressively move the window to the right
        inline void MovingProduct(
            const int step, 
            const Eigen::Matrix<float, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& mat1,
            Eigen::Map<const Matrix>& mat2, Matrix& res)
        {
            const int row1 = mat1.rows();
            const int col1 = mat1.cols();
            const int row2 = mat2.rows();
            const int col2 = mat2.cols();
            const int colEnd = col1-row2;
            int startCol = 0;
            for (int leftEnd = 0; leftEnd <= colEnd; leftEnd += step, startCol += col2)
            {
                res.block(0, startCol,row1,col2).noalias() += mat1.block(0,leftEnd,row1,row2)*mat2;
            }
            
        }

        // The main convolution function using the "valid" rule
        inline void Convolve_Valid(
            const ConvDims& dim,
            const float* src, const bool imageOuterLoop, const int nObs,
            const float* filterData, float* dest
        )
        {
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrix;
            typedef Eigen::Map<const Matrix> ConstMapMat;
            //flat Matrix
            const int flatRows = dim.ConvRows * nObs;
            const int flatCols = dim.FilterRows * dim.ChannelCols;
            const int channelSize = dim.ChannelRows * dim.ChannelCols;
            //distance between two images
            const int ImgStride = imageOuterLoop ? (dim.ImgRows * dim.ImgCols) : channelSize;
            //distance between two channels
            const int ChannelStride = imageOuterLoop ? channelSize : (channelSize*nObs);
            RMatrix flatMat(flatRows,flatCols);

            const int& resRows = flatRows;
            const int resCols = dim.ConvCols * dim.outChannels;
            Matrix res = Matrix::Zero(resRows,resCols);
            const int& step = dim.FilterRows;
            const int filterSize = dim.FilterRows*dim.FilterCols;
            const int filterStride = filterSize * dim.outChannels;

            for (int i = 0; i < dim.inChannels; i++, src += ChannelStride, filterData += filterStride)
            {
                //flatten img
                FlattenMat(dim,src,ImgStride,nObs,flatMat);
                //convolve
                ConstMapMat filter(filterData, filterSize, dim.outChannels);
                MovingProduct(step, flatMat, filter, res);
            }
            // The layout of 'res' is very complicated
            /*
            * obs0_out0[0, 0] obs0_out1[0, 0] obs0_out2[0, 0] obs0_out0[0, 1] obs0_out1[0, 1] obs0_out2[0, 1] ...
            * obs0_out0[1, 0] obs0_out1[1, 0] obs0_out2[1, 0] obs0_out0[1, 1] obs0_out1[1, 1] obs0_out2[1, 1] ...
            * obs0_out0[2, 0] obs0_out1[2, 0] obs0_out2[2, 0] obs0_out0[2, 1] obs0_out1[2, 1] obs0_out2[2, 1] ...
            * obs1_out0[0, 0] obs1_out1[0, 0] obs1_out2[0, 0] obs1_out0[0, 1] obs1_out1[0, 1] obs1_out2[0, 1] ...
            * obs1_out0[1, 0] obs1_out1[1, 0] obs1_out2[1, 0] obs1_out0[1, 1] obs1_out1[1, 1] obs1_out2[1, 1] ...
            * obs1_out0[2, 0] obs1_out1[2, 0] obs1_out2[2, 0] obs1_out0[2, 1] obs1_out1[2, 1] obs1_out2[2, 1] ...
            * ...
            *
            */
           // obs<k>_out<l> means the convolution result of the k-th image on the l-th output channel
            // [i, j] gives the matrix indices
            // The destination has the layout
            /*
            * obs0_out0[0, 0] obs0_out0[0, 1] obs0_out0[0, 2] obs0_out1[0, 0] obs0_out1[0, 1] obs0_out1[0, 2] ...
            * obs0_out0[1, 0] obs0_out0[1, 1] obs0_out0[1, 2] obs0_out1[1, 0] obs0_out1[1, 1] obs0_out1[1, 2] ...
            * obs0_out0[2, 0] obs0_out0[2, 1] obs0_out0[2, 2] obs0_out1[2, 0] obs0_out1[2, 1] obs0_out1[2, 2] ...
            *
            */
            // which in a larger scale looks like
            // [obs0_out0 obs0_out1 obs0_out2 obs1_out0 obs1_out1 obs1_out2 obs2_out0 ...]
            const int destRows = dim.ChannelRows;
            const int destCols = resCols*nObs;
            const float* resData = res.data();
            const std::size_t copyBytes = sizeof(float)*destRows;
            for (int b = 0; b < destCols; b++, dest += destRows)
            {
                const int k = b / resCols;
                const int l = (b%resCols)/dim.ConvCols;
                const int j = b % dim.ConvCols;
                const int d = j * dim.outChannels + l;
                const int resColHead = d * resRows;
                std::memcpy(dest,resData+resColHead+k*dim.ConvRows,copyBytes);
            }
            
        }
    } // namespace internal
    
} // namespace MiniBrain
