#ifndef KERNELS_HPP
#define KERNELS_HPP

/*Header file that interfaces the CUDA kernels with main program*/

// Template structure to pass to kernel
template <typename T>
struct KernelArray
{
    T*  _array;
    int _size;
};

// Function to convert device_vector to structure
template <typename T>
KernelArray<T> convertToKernel(thrust::device_vector<T>& dVec)
{
    KernelArray<T> kArray;
    kArray._array = thrust::raw_pointer_cast(&dVec[0]);
    kArray._size  = (int) dVec.size();
 
    return kArray;
}

extern "C" void kernel_WarmUp();
extern "C" void Wrap_RotatePerm(uint8_t * in, uint8_t * out, int* colRotate, int* rowRotate, const dim3 & grid, const dim3 & block, const int mode);
extern "C" void Wrap_Diffusion(uint8_t * &in, uint8_t * &out, const double*& randRowX, const double*& randRowY, const int dim[], const double r, const int mode);



#endif
