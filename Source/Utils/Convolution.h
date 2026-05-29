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
        
        inline void FlattenMat(const ConvDims& Dim, const Scalar* Src, const int stride, const int nObs,
            Eigen::Matrix<Scalar, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& FlatMat)
        {
            const int& SegmentSize = Dim.FilterRows;
            const std::size_t copyBytes = sizeof(Scalar)*SegmentSize;
            Scalar* Writer = FlatMat.data();
            const int ChannelSize = Dim.ChannelRows * Dim.ChannelCols;

            for (int i = 0; i < nObs; i++, Src += stride)
            {
                const Scalar* ReaderRow = Src;
                const Scalar* const ReaderRowEnd = Src+Dim.ConvRows;
                for ( ; ReaderRow < ReaderRowEnd; ReaderRow++)
                {
                    const Scalar* Reader = ReaderRow;
                    const Scalar* const ReaderEnd = Reader + ChannelSize;

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
            const Eigen::Matrix<Scalar, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& mat1,
            Eigen::Map<const Matrix<Scalar>>& mat2, Matrix<Scalar>& res)
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
            const Scalar* src, const bool imageOuterLoop, const int nObs,
            const Scalar* filterData, Scalar* dest
        )
        {
            typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrix;
            typedef Eigen::Map<const Matrix<Scalar>> ConstMapMat;
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
            Matrix<Scalar> res = Matrix<Scalar>::Zero(resRows,resCols);
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
            const int destRows = dim.ConvRows;
            const int destCols = resCols*nObs;
            const Scalar* resData = res.data();
            const std::size_t copyBytes = sizeof(Scalar)*destRows;
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

        // 专门给 AutoDiff 训练模式使用的前向卷积工具函数
        template <typename T>
        void Convolve_Valid_Autodiff(
            const ConvDims& dim,
            const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& InData,       // 输入改为模板 T
            const int nObs,
            const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& filterData,   // 权重改为模板 T
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& dest               // 输出改为模板 T
        )
        {
            dest.setZero();

            const int outRows = dim.ConvRows;
            const int outCols = dim.ConvCols;
            const int outCells = outRows * outCols;
            const int filterCells = dim.FilterRows * dim.FilterCols;
            const int imgCells = dim.ImgRows * dim.ImgCols;

            // ==========================================
            // 1. 按照原版一维规律，重组权重矩阵 W (内部全部用 T)
            // ==========================================
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> W(dim.outChannels, dim.inChannels * filterCells);
            const int filterStride = filterCells * dim.outChannels;

            for (int ic = 0; ic < dim.inChannels; ++ic) {
                int ic_offset = ic * filterStride;
                for (int oc = 0; oc < dim.outChannels; ++oc) {
                    int oc_offset = ic_offset + oc * filterCells;
                    // 使用 block 建立 T 类型的表达式树依赖
                    W.block(oc, ic * filterCells, 1, filterCells) = 
                        filterData.block(oc_offset, 0, filterCells, 1).transpose();
                }
            }

            // ==========================================
            // 2. 针对每个样本构建临时的 Im2Col 矩阵 (内部全部用 T)
            // ==========================================
            for (int o = 0; o < nObs; ++o) 
            {
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X_col(dim.inChannels * filterCells, outCells);

                int col_idx = 0;
                for (int r = 0; r < outRows; ++r) {
                    for (int c = 0; c < outCols; ++c) {
                        int row_idx = 0;
                        for (int ic = 0; ic < dim.inChannels; ++ic) {
                            int img_channel_start = ic * imgCells;
                            
                            for (int fr = 0; fr < dim.FilterRows; ++fr) {
                                for (int fc = 0; fc < dim.FilterCols; ++fc) {
                                    int img_r = r + fr;
                                    int img_c = c + fc;
                                    int img_pixel = img_channel_start + (img_r * dim.ImgCols + img_c);
                                    
                                    // 这里会将 InData 中的 varf/var 正确赋值给 X_col，保留计算图
                                    X_col(row_idx++, col_idx) = InData(img_pixel, o);
                                }
                            }
                        }
                        col_idx++;
                    }
                }

                // ==========================================
                // 3. 核心大矩阵乘法 Y = W * X_col
                // 此时 Eigen 会自动调用针对 autodiff::varf 重载的 operator*
                // 这将在内存中生成极其高效的矩阵求导图节点
                // ==========================================
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y = W * X_col; 

                // 4. 将结果无缝还原到输出矩阵 dest 对应的位置上
                for (int oc = 0; oc < dim.outChannels; ++oc) {
                    dest.block(oc * outCells, o, outCells, 1) = Y.block(oc, 0, 1, outCells).transpose();
                }
            }
        }

        //这是手工反向传播的工具函数，已经换用自动微分了
        // The moving_product() function for the "full" rule
        /*inline void MovingProduct_Full(
            const int padding, const int step,
            const Eigen::Matrix<Scalar,  Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>& mat1,
            const Matrix& mat2,
            Matrix& res
        )
        {
            const int row1 = mat1.rows();
            const int col1 = mat1.cols();
            const int row2 = mat2.rows();
            const int col2 = mat2.cols();
            int resStartCol = 0;

            int leftEnd = -padding;
            int rightEnd = step;

            for(; leftEnd < 0 && rightEnd <= col1; leftEnd += step, rightEnd += step, resStartCol += col2)
            {
                res.block(0, resStartCol, row1, col2).noalias() += mat1.leftCols(rightEnd) * mat2.bottomRows(rightEnd);
            }

            //main
            for ( ; rightEnd <= col1; leftEnd += step, rightEnd += step, resStartCol += col2)
            {
                res.block(0, resStartCol, row1, col2).noalias() += mat1.block(0, leftEnd, row1, row2) * mat2;
            }
            
            //right padding
            for (; leftEnd < col1; leftEnd += step, resStartCol += col2 )
            {
                if (leftEnd <= 0)
                {
                    res.block(0, resStartCol, row1, col2).noalias() += mat1 * mat2.block(0, - leftEnd, col1, row2);
                }
                else
                {
                    const int overlap = col1 - leftEnd;
                    res.block(0, resStartCol, row1, col2).noalias() += mat1.rightCols(overlap) * mat2.topRows(overlap);
                }
            }
            
        }*/

        //有自动微分之后这个反向传播的函数就不需要了，先注释掉
        // The main convolution function for the "full" rule 反向传播要用到
        /*inline void Convolve_Full(const ConvDims& dim, const Scalar* src, const int nObs, const Scalar* filterData, Scalar* dest)
        {
            typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatrix;
            typedef Eigen::Map<const Matrix> ConstMapMat;

            const int paddingTop = dim.FilterRows - 1;
            const int paddingLeft = dim.FilterCols - 1;

            const int convRows = dim.ChannelRows + paddingTop;
            const int convCols = dim.ChannelCols + paddingLeft;

            const int padRows = dim.ImgRows + paddingTop * 2;
            const int padCols = dim.ImgCols * nObs;

            Matrix padMat(padRows, padCols);
            ConstMapMat srcMat(src, dim.ImgRows, padCols);

            padMat.topRows(paddingTop).setZero();
            padMat.bottomRows(paddingTop).setZero();
            padMat.block(paddingTop, 0, dim.ImgRows, padCols).noalias() = srcMat;
            src = padMat.data();
            ConvDims padDim(dim.inChannels,dim.outChannels,padRows,dim.ChannelCols,dim.FilterRows,dim.FilterCols);

            const int flatRows = convRows * nObs;
            const int flatCols = dim.FilterRows * dim.ChannelCols;
            const int imgStride = padRows * dim.ImgCols;
            const int channelStride = padRows * dim.ChannelCols;
            RMatrix flatMat(flatRows,flatCols);
            // The processing of filters are different from the "valid" rule in two ways:
            // 1. The layout of input channels and output channels are switched
            // 2. The filters need to be rotated, which is equivalent to reversing the vector of each filter
            // We also separate filters that belong to different input channels
            std::vector<Matrix> filtersIn(dim.inChannels);
            const int filterSize = dim.FilterRows * dim.FilterCols;
            const int nFilter = dim.inChannels * dim.outChannels;

            for (int i = 0; i < dim.inChannels; i++)
            {
                filtersIn[i].resize(filterSize,dim.outChannels);
            }

            const Scalar* reader = filterData;

            for (int i = 0; i < nFilter; i++, reader += filterSize)
            {
                Scalar* writer = filtersIn[i % dim.inChannels].data() + (i/dim.inChannels)*filterSize;
                std::reverse_copy(reader, reader + filterSize, writer);
            }
            
            //results
            const int& resRows = flatRows;
            const int resCols = convCols * dim.outChannels;
            Matrix res = Matrix::Zero(resRows,resCols);
            const int& step = dim.FilterRows;
            const int filterPadding = paddingLeft * dim.FilterRows;

            for (int i = 0; i < dim.inChannels; i++, src += channelStride)
            {
                FlattenMat(padDim, src, imgStride, nObs, flatMat);
                MovingProduct_Full(filterPadding, step, flatMat, filtersIn[i], res);
            }
            
            const int& destRows = convRows;
            const int destCols = resCols * nObs;
            const Scalar* resData = res.data();
            const std::size_t copyBytes = sizeof(Scalar) * destRows;

            for(int b = 0; b < destCols; b++, dest += destRows)
            {
                const int k = b/resCols;
                const int l = (b % resCols)/convCols;
                const int j = b%convCols;
                const int d = j* dim.outChannels + l;
                const int resCloHead = d * resRows;
                std::memcpy(dest,resData+resCloHead+k*convRows,copyBytes);
            }
        }*/
    } // namespace internal
    
} // namespace MiniBrain
