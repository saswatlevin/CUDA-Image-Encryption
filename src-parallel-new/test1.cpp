#include <iostream>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdbool>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "kernels.hpp"

#define DEBUG_VECTORS 0

using namespace cv;
using namespace std;
using namespace chrono;
using namespace thrust;

/*Prototypes*/
static inline void printImageContents(cv::Mat image, uint32_t channels);
inline void twodLM(double &x, double &y, const double &r);
static inline int getRandomInteger(int lower_bound,int upper_bound);
static inline double getRandomDouble(double lower_limit,double upper_limit);
int* genRandVec(const int m,const int n,int skip_offset,double X,double Y,double R);
static inline void printArray8(uint8_t *arr,int length);

static inline void printArray8(uint8_t *arr,int length)
{
  for(int i = 0; i < length; ++i)
  {
    printf(" %d",arr[i]);
  }
}

static inline void printArrayInt(int *arr,int length)
{
  for(int i = 0; i < length; ++i)
  {
    printf(" %d",arr[i]);
  }
}

static inline void printImageContents(cv::Mat image,uint32_t channels)
{
  for(uint32_t i=0;i<image.rows;++i)
  { 
    printf("\n");
    for(uint32_t j=0;j<image.cols;++j)
    {
       for(uint32_t k=0;k < channels;++k)
       {
          
        printf("%d\t",image.at<Vec3b>(i,j)[k]); 
       } 
     }
  }
}


inline void twodLM(double &x, double &y, const double &r)
{
  x = r * (3 * y + 1) * x * (1 - x);
  y = r * (3 * x + 1) * y * (1 - y);
  //x = fmod(x,1);
  //printf(" %f",x);
  return; 
} 

static inline int getRandomInteger(int lower_bound,int upper_bound)
{
  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
  mt19937 seeder(seed);
  uniform_int_distribution<int> intGen(lower_bound, upper_bound);
  int alpha = intGen(seeder);
  return alpha;
}

static inline double getRandomDouble(double lower_limit,double upper_limit)
{
   std::random_device r;
   std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
   mt19937 seeder(seed);
   uniform_real_distribution<double> realGen(lower_limit, upper_limit);   
   auto randnum=realGen(seeder);
   return (double)randnum;
}

int* genRandVec(const int m,const int n,int skip_offset,double X,double Y,double R)
{
  
  int exp = (int)pow(10,9);

  thrust::host_vector<int> host_random_vector(m);
  
  auto start_rand = high_resolution_clock::now();
  for(int i = 0; i < skip_offset; ++i)
  {
    twodLM(X,Y,R);
  }
  
  for(int i = 0; i < m; ++i)
  {
    twodLM(X,Y,R);
    host_random_vector[i] = (int)(X * exp) % n;
  }
  auto end_rand = high_resolution_clock::now();
  
  auto duration_rand = duration_cast<microseconds>(end_rand - start_rand).count();
  cout<<"\nduration_rand = "<<duration_rand<<" us";
  thrust::device_vector<int> device_random_vector = host_random_vector;
  int *device_random_vector_pointer = thrust::raw_pointer_cast(&host_random_vector[0]);
  return device_random_vector_pointer;
}

int main()
{
  int i = 0,rounds = 0;
  
  cv::Mat image;
  std::string image_path;
  cout<<"\nEnter image path\t";
  cin>>image_path;
  cout<<"\nEnter the number of rotation rounds\t";
  cin>>rounds;
  
  auto start_read = high_resolution_clock::now();
  
  image = cv::imread(image_path,cv::IMREAD_UNCHANGED);
  
  auto end_read = high_resolution_clock::now();
  
  auto read_duration = duration_cast<microseconds>(end_read - start_read).count();
  cout<<"\nRead duration = "<<read_duration<<" us";
  
  if(!image.data)
  {
    cout<<"\nCould not open image\nExiting...";
    exit(0);
  }
  
  //cv::resize(image,image,cv::Size(50,50),CV_INTER_LANCZOS4);
  
  int m = (int)image.rows;
  int n = (int)image.cols;
  int channels = (int)image.channels();
  int total = m * n * channels;
  int skip_offset = getRandomInteger(10,40);
  size_t img_size = total * sizeof(uint8_t);
  
  
  auto start_pointer_declaration = high_resolution_clock::now();
  
  uint8_t *img_in = (uint8_t*)calloc(total,sizeof(uint8_t));
  uint8_t *img_out = (uint8_t*)calloc(total,sizeof(uint8_t));
  int *u = (int*)calloc(n,sizeof(int));
  int *v = (int*)calloc(m,sizeof(int));
  
  uint8_t *gpu_img_in; 
  uint8_t *gpu_img_out;
  int *gpu_u;
  int *gpu_v;
  
  auto end_pointer_declaration = high_resolution_clock::now();
  
  auto duration_pointer_declaration = duration_cast<microseconds>(end_pointer_declaration - start_pointer_declaration).count();
  cout<<"\nPointer declaration duration = "<<duration_pointer_declaration<<" us";
  
  //Copy plain image data to img_in 
  auto start_copy = high_resolution_clock::now();
  std::memcpy(img_in,image.data,total * sizeof(uint8_t)); 
  auto end_copy = high_resolution_clock::now();
  auto copy_duration = duration_cast<microseconds>(end_copy - start_copy).count();
  cout<<"\nCopying image data duration = "<<copy_duration <<" us";
  
  if(DEBUG_VECTORS == 1)
  {
    cout<<"\nOriginal image = \n";
    printArray8(img_in,total);
  }
  
  cudaMalloc((void**)&gpu_img_in, total * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_img_out, total * sizeof(uint8_t));
  cudaMalloc((void**)&gpu_u, n * sizeof(int));
  cudaMalloc((void**)&gpu_v, m * sizeof(int));
  
  cudaMemcpy(gpu_img_in, img_in, total * sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_img_out, img_out, total * sizeof(uint8_t), cudaMemcpyHostToDevice);
  
  
  const dim3 grid(m, n, 1);
  const dim3 block(channels, 1, 1);
  /*Generate U and V*/
  double X = getRandomDouble(0.1,0.9);
  double Y = getRandomDouble(0.1,0.9);
  double R = getRandomDouble(1.11,1.19);
  
  u = genRandVec(m,n,skip_offset,X,Y,R);
  v = genRandVec(n,m,skip_offset,X,Y,R);
  
  /*Copy U and V from CPU memory to GPU memory*/
  cudaMemcpy(gpu_u, u, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_v, v, m * sizeof(int), cudaMemcpyHostToDevice);      
  
  cout<<"\n-------------Encryption--------------\n";
  
  for(int i = 0; i < rounds; ++i)
  {
    cout<<"\n=============ROUND================= "<<i + 1;
    
    auto start_perm = high_resolution_clock::now(); 
    
    Wrap_RotatePerm(gpu_img_in,gpu_img_out,gpu_u,gpu_v,grid,block,0);
    
    auto end_perm = high_resolution_clock::now();
    
    auto perm_duration = duration_cast<microseconds>(end_perm - start_perm).count();
    cout<<"\nPermutation kernel duration = "<<perm_duration<<" us";
    
    
    
    
    auto start_getresult = high_resolution_clock::now();
    
    //Copying result from GPU memory to Host memory
    cudaMemcpy(img_in, gpu_img_in, total * sizeof(uint8_t), cudaMemcpyDeviceToHost);      
    
    auto end_getresult = high_resolution_clock::now();
    
    auto getresult_duration = duration_cast<microseconds>(end_getresult - start_getresult).count();
    
    cout<<"\ncudaMemcpy device to host duration = "<<getresult_duration<<" us"; 
    
    auto start_memcpy = high_resolution_clock::now();
     
    //Copying result back to Mat image
    std::memcpy(image.data, img_in, total * sizeof(uint8_t));
    
    auto end_memcpy = high_resolution_clock::now();
    
    auto memcpy_duration = duration_cast<microseconds>(end_memcpy - start_memcpy).count();
    
    cout<<"\nImage vector to image matrix copy duration = "<<memcpy_duration<<" us";
    
    if(DEBUG_VECTORS == 1)
    {
      cout<<"\n\nimg_in = \n";
      printArray8(img_in,total);
      //cout<<"\n\nimg_out = \n";
      //printArray8(img_out,total);
      cout<<"\n\nu = \n";
      printArrayInt(u,n);
      cout<<"\n\nv = \n";
      printArrayInt(v,m);
      
    }
  
    //Writing intermediate encrypted image
    image_path = "_encrypted_" + std::to_string(i) + "_" + ".png";
    
      
    auto start_write = high_resolution_clock::now();  
    
    cv::imwrite(image_path,image);
    
    auto end_write = high_resolution_clock::now();
    
    auto write_duration = duration_cast<microseconds>(end_write - start_write).count();
      
    cout<<"\nEncrypted image write duration = "<<write_duration<<" us";
    image_path =  "";
   
  }  
  
  cout<<"\n-------------Decryption--------------\n";
  for(int i = rounds - 1; i >= 0; --i)
  {
    cout<<"\n=============ROUND================= "<<i + 1;
    
    auto start_unperm = high_resolution_clock::now(); 
    
    Wrap_RotatePerm(gpu_img_in,gpu_img_out,gpu_u,gpu_v,grid,block,1);
    
    auto end_unperm = high_resolution_clock::now();
    
    auto unperm_duration = duration_cast<microseconds>(end_unperm - start_unperm).count();
    cout<<"\nUnpermutation kernel duration = "<<unperm_duration<<" us";
    
    //Copying result from GPU memory to Host memory
 
    auto start_getresult = high_resolution_clock::now();
    
    cudaMemcpy(img_in, gpu_img_in, total * sizeof(uint8_t), cudaMemcpyDeviceToHost);    
    
    auto end_getresult = high_resolution_clock::now(); 
    auto getresult_duration = duration_cast<microseconds>(end_getresult - start_getresult).count();
    cout<<"\ncudaMemcpy device to host duration  = "<<getresult_duration<<" us";
    
    if(DEBUG_VECTORS == 1)
    {
      cout<<"\nimg_in = \n";
      printArray8(img_in,total);
      //cout<<"\nimg_out = \n";
      //printArray8(img_out,total);
      cout<<"\n\nu = \n";
      printArrayInt(u,n);
      cout<<"\n\nv = \n";
      printArrayInt(v,m);
    }
    
    
    //Copying result back to Mat image
    auto start_memcpy = high_resolution_clock::now();
    
    std::memcpy(image.data, img_in, total * sizeof(uint8_t));
    
    auto end_memcpy = high_resolution_clock::now();
    
    auto memcpy_duration = duration_cast<microseconds>(end_memcpy - start_memcpy).count();
    
    cout<<"\nImage vector to image matrix copy duration = "<<memcpy_duration<<" us";
      
    //Writing intermediate encrypted image
    image_path = "_decrypted_" + std::to_string(i) + "_" + ".png";
    
    auto start_write = high_resolution_clock::now();  
    
    cv::imwrite(image_path,image);
    
    auto end_write = high_resolution_clock::now();
    
    auto write_duration = duration_cast<microseconds>(end_write - start_write).count();
      
    cout<<"\nDECRYPTED IMAGE WRITE DURATION = "<<write_duration<<" us";
    image_path =  "";
      
   }    

  return 0;
}


