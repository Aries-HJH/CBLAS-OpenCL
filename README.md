# CBLAS-OpenCL
使用OpenCL实现CBLAS函数库

其中复数使用的结构是：
```
typedef struct {  
    double real; // 实部  
    double imag; // 虚部  
} dComplex; 

typedef struct {  
    float real; // 实部  
    float imag; // 虚部  
} fComplex; 
```
目前进度：
```
    cblas_sswap
    cblas_scopy
    cblas_saxpy
    cblas_dswap
    cblas_dcopy
    cblas_daxpy
    cblas_cswap
    cblas_ccopy
    cblas_caxpy
    cblas_zswap
    cblas_zcopy
    cblas_zaxpy
    cblas_sdsdot
    cblas_sdot
    cblas_ddot
    cblas_dsdot
    cblas_cdotu_sub
    cblas_cdotc_sub
    cblas_zdotu_sub
    cblas_zdotc_sub
    cblas_cscal
    cblas_dscal
    cblas_sscal
    cblas_zscal
    cblas_zdscal
    cblas_csscal
    cblas_srotg
```