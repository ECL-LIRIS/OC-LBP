// Author: Chao Zhu (LIRIS, Ecole Centrale de Lyon)
// Email: chao.zhu@ec-lyon.fr
// This is a part of the OC-LBP descriptor codes.

// Reference:
// C. Zhu, C.-E. Bichot and L. Chen, Image region description using orthogonal combination of local binary patterns
// enhanced with color information, Pattern Recognition (2013), http://dx.doi.org/10.1016/j.patcog.2013.01.003

#include "mex.h"
#include <math.h>
#define THRES 0.2

inline void computeLBPonKeypoints(const double* lbp_image,int height,int width,int x,int y,int dims_lbp,int offset,int num,double* descs_out);
inline void computeHistogram(const double* lbp_image,int height,int xstart,int xend,int ystart,int yend,double* lbp_histogram);
inline void normalizeHistogram(double* lbp_histogram,int dims);
inline void clipHistogram(double* lbp_histogram,int dims,double thres);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Check input and output */
    if (nrhs != 5)
        mexErrMsgTxt("Must have 5 input arguments!");
    if (nlhs != 1)
        mexErrMsgTxt("Must have 1 output argument!");
    if (!mxIsDouble(prhs[0])||!mxIsDouble(prhs[1])||!mxIsDouble(prhs[2])||!mxIsDouble(prhs[3])||!mxIsDouble(prhs[4]))
        mexErrMsgTxt("All input arguments must be double class!");
    
    /* Get LBP coded image data */
    const double *lbp_image;
    lbp_image = mxGetPr(prhs[0]);
    int height = mxGetM(prhs[0]);
    int width = mxGetN(prhs[0]);
    
    /* Get keypoint frames */
    const double *frm;
    frm = mxGetPr(prhs[1]);
    
    /* Get dimension of LBP histogram */
    const double dims_lbp = mxGetScalar(prhs[2]);
    
    /* Get coordinate offset for square patch */
    const double offset = mxGetScalar(prhs[3]);
    
    /* Get num-by-num cells for square patch */
    const double num = mxGetScalar(prhs[4]);
    
    /* Output */
    int num_keyps = mxGetN(prhs[1]);
    int dims = (int)dims_lbp*(int)num*(int)num;
    
    double *descs;
    plhs[0] = mxCreateDoubleMatrix(dims,num_keyps,mxREAL);
    descs = mxGetPr(plhs[0]);
    
    for (int i=0;i<num_keyps;i++)
    {
        int x,y;
        x = (int)frm[i*2];
        y = (int)frm[i*2+1];
        computeLBPonKeypoints(lbp_image,height,width,x-1,y-1,(int)dims_lbp,(int)offset,(int)num,descs+i*dims);
    }
}

inline void computeLBPonKeypoints(const double* lbp_image,int height,int width,int x,int y,int dims_lbp,int offset,int num,double* descs_out)
{
    /* Get square neighboring area around the keypoint */
    int xmin = x - offset;
    int xmax = x + offset;
    int ymin = y - offset;
    int ymax = y + offset;
    
    if (xmin < 0)
        xmin = 0;
    if (xmax > width-1)
        xmax = width-1;
    if (ymin < 0)
        ymin = 0;
    if (ymax > height-1)
        ymax = height-1;
    
    /* Divide square patch into cells and compute LBP histogram within each cell */
    int cell_width,cell_height;
    int square_width = xmax-xmin+1;
    int square_height = ymax-ymin+1;
    
    cell_width = floor((double)square_width/(double)num);
    cell_height = floor((double)square_height/(double)num);
    
    for (int i=0;i<num;i++)
        for (int j=0;j<num;j++)
        {
            int xstart = xmin + i*cell_width;
            int xend = xstart + cell_width -1;
            int ystart = ymin + j*cell_height;
            int yend = ystart + cell_height -1;
            
            computeHistogram(lbp_image,height,xstart,xend,ystart,yend,descs_out+dims_lbp*(i*num+j));
        }
    
    int dims = dims_lbp*num*num;
    normalizeHistogram(descs_out,dims);
    clipHistogram(descs_out,dims,THRES);
    normalizeHistogram(descs_out,dims);
}

inline void computeHistogram(const double* lbp_image,int height,int xstart,int xend,int ystart,int yend,double* lbp_histogram)
{
    for (int i=xstart;i<=xend;i++)
        for (int j=ystart;j<=yend;j++)
        {
            int ind,value;
            
            ind = height*i+j;
            value = (int)lbp_image[ind];
            
            ++lbp_histogram[value];
        }
}

inline void normalizeHistogram(double* lbp_histogram,int dims)
{
    double sum = 0.0;
    
    for (int i=0;i<dims;i++)
        sum += lbp_histogram[i] * lbp_histogram[i];
    
    if (sum != 0.0)
    {
        sum = sqrt(sum);
        for (int i=0;i<dims;i++)
            lbp_histogram[i] /= sum;
    }
}

inline void clipHistogram(double* lbp_histogram,int dims,double thres)
{
    for (int i=0;i<dims;i++)
        if (lbp_histogram[i] > thres)
            lbp_histogram[i] = thres;
}
