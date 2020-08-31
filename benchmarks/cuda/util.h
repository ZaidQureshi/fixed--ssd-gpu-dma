#ifndef __UTIL_H__
#define __UTIL_H__

#include <cuda.h>

#define cuda_err_chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess)
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(1);
    }
}

#endif // __UTIL_H__
