#include <iostream> /*For IO*/
#include <cstdio>   /*For printf()*/
#include <cstdlib>  /*For malloc()*/
#include <cstdint>  /*For standard variable support*/
#include <opencv2/opencv.hpp> /*For OpenCV*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include "include/randomfunctions.hpp"
#include "include/selfxorfunctions.hpp"

void rowColLUTGen(uint32_t *&colSwapLUT,uint32_t *&colRandVec,uint32_t *&rowSwapLUT,uint32_t *&rowRandVec,uint32_t n);
void genLUTVec(uint32_t *&lutVec,uint32_t n);
void writeVectorToFile32(uint32_t *&vec,int length,std::string filename);
void writeVectorToFile8(uint8_t *&vec,int length,std::string filename);

void genLUTVec(uint32_t *&lutVec,uint32_t n)
{
  for(int i = 0; i < n; ++i)
  {
    lutVec[i]=i;
  }
}

void rowColLUTGen(uint32_t *&colSwapLUT,uint32_t *&colRandVec,uint32_t *&rowSwapLUT,uint32_t *&rowRandVec,uint32_t n)
{
  int jCol=0,jRow=0;
  for(int i = n - 1; i > 0; i--)
  {
    jCol = colRandVec[i] % i;
    std::swap(colSwapLUT[i],colSwapLUT[jCol]);
  } 
  
  for(int i = n - 1; i > 0; i--)
  {
    jRow = rowRandVec[i] % i;
    std::swap(rowSwapLUT[i],rowSwapLUT[jRow]);
  } 
}

void writeVectorToFile32(uint32_t *&vec,int length,std::string filename)
{
  std::ofstream file(filename);
  if(!file)
  {
    cout<<"\nCould not create "<<filename<<"\nExiting...";
    exit(0);
  }

  std::string elements = std::string("");  

  for(int i = 0; i < length; ++i)
  {
    elements.append(std::to_string(vec[i]));
    elements.append("\n");
  }
  file<<elements;
  file.close();
}

void writeVectorToFile8(uint8_t *&vec,int length,std::string filename)
{
  std::ofstream file(filename);
  if(!file)
  {
    cout<<"\nCould not create "<<filename<<"\nExiting...";
    exit(0);
  }
  
  std::string elements = std::string("");
  for(int i = 0; i < length; ++i)
  {
    elements.append(std::to_string(vec[i]));
    elements.append("\n");
  }
  
  file<<elements;
  file.close();
}

int main()
{
  cv::Mat image;
  uint32_t m=0,n=0,total=0;
  int lowerLimit=0,upperLimit=0;

  image = cv::imread("airplane.png",IMREAD_COLOR);
  if(!image.data)
  {
    cout<<"\nImage not found \nExiting...";
    exit(0);  
  }
  
  if(RESIZE_TO_DEBUG==1)
  {
    cv::resize(image,image,cv::Size(512,512));
  }
  
  if(PRINT_IMAGES == 1)
  {
    cout<<"\nOriginal image = ";
    printImageContents(image);
  }
  
  m = (uint32_t)image.rows;
  n = (uint32_t)image.cols;
  total = m * n;
  
  cout<<"\nRows = "<<m;
  cout<<"\nColumns = "<<n;
  cout<<"\nChannels = "<<image.channels();
  cout<<"\nTotal = "<<total;
  
  Mat img_enc = Mat(Size(m, n), CV_8UC3, Scalar(0, 0, 0));
  Mat img_dec = Mat(Size(m, n), CV_8UC3, Scalar(0, 0, 0)); 
  
  cout<<"\nempty_image rows = "<<img_enc.rows;
  cout<<"\nempty_image columns = "<<img_dec.cols;
  
  
  uint32_t *colSwapLUT = (uint32_t*)malloc(sizeof(uint32_t) * n);
  uint32_t *rowSwapLUT = (uint32_t*)malloc(sizeof(uint32_t) * n);
  uint32_t *rowRandVec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint32_t *colRandVec = (uint32_t*)malloc(sizeof(uint32_t) * total * 3);
  uint8_t *img_vec = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);
  uint8_t *enc_vec = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);
  uint8_t *dec_vec = (uint8_t*)malloc(sizeof(uint8_t) * total * 3);
  
  flattenImage(image,img_vec);  

  lowerLimit = 1;
  upperLimit = total * 3; 

  genLUTVec(colSwapLUT,n);
  genLUTVec(rowSwapLUT,n);
  
  if(DEBUG_VECTORS==1)
  {
    /*cout<<"\ncolSwapLUT before swap = ";
    for(int i = 0 ;i < n; ++i)
    {
      printf(" %d",colSwapLUT[i]);
    }
    
    cout<<"\nrowSwapLUT before swap = ";
    for(int i = 0; i < n; ++i)
    {
      printf(" %d",rowSwapLUT[i]);
    }*/
   
  } 
 
  
  MTMap(rowRandVec,total,lowerLimit,upperLimit);
  MTMap(colRandVec,total,lowerLimit,upperLimit);
  
  if(DEBUG_VECTORS == 1)
  {
    /*cout<<"\nrowRandVec = ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf("\n %d",rowRandVec[i]);
    }
    
    cout<<"\ncolRandVec = ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf("\n%d",colRandVec[i]);
    }*/
    
     writeVectorToFile32(rowRandVec,total * 3,"Reports/rowRandVec260.txt");
     writeVectorToFile32(colRandVec,total * 3,"Reports/colRandVec260.txt"); 
  }
  
  rowColLUTGen(colSwapLUT,colRandVec,rowSwapLUT,rowRandVec,n);
  
  if(DEBUG_VECTORS == 1)
  {
    /*cout<<"\ncolSwapLUT after swap = ";
    for(int i = 0 ;i < n; ++i)
    {
      printf(" %d",colSwapLUT[i]);
    }
    
    cout<<"\nrowSwapLUT after swap = ";
    for(int i = 0; i < n; ++i)
    {
      printf(" %d",rowSwapLUT[i]);
    }*/
    
    writeVectorToFile32(rowSwapLUT,n,"Reports/rowSwap260.txt");
    writeVectorToFile32(colSwapLUT,n,"Reports/colswap260.txt");
  }  
  
  
  rowColSwapEnc(img_vec,enc_vec,rowSwapLUT,colSwapLUT,m,n,total);
  if(PRINT_IMAGES == 1)
  {
    cout<<"\nempty_image after encryption = ";
    printImageContents(img_enc);
    
    
  }  
  
  if(DEBUG_VECTORS == 1)
  {
    /*cout<<"\n\nOriginal image = ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",img_vec[i]);
    }
    
    cout<<"\n\nEncrypted image = ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",enc_vec[i]);
    }*/
    writeVectorToFile8(img_vec,total * 3,"Reports/img_vec260.txt");
    writeVectorToFile8(enc_vec,total * 3,"Reports/enc_vec260.txt");
  }  

  if(DEBUG_IMAGES == 1)
  {
    cv::Mat img_reshape(m,n,CV_8UC3,enc_vec);
    cv::imwrite("airplane_rowcolswap_encrypted.png",img_reshape);
  }
  
  rowColSwapDec(enc_vec,dec_vec,rowSwapLUT,colSwapLUT,m,n,total);
  if(DEBUG_VECTORS == 1)
  {
    /*cout<<"\n\nDecrypted image = ";
    for(int i = 0; i < total * 3; ++i)
    {
      printf(" %d",dec_vec[i]);
    }*/
    writeVectorToFile8(dec_vec,total * 3,"Reports/dec_vec260.txt");
  }
  
  if(PRINT_IMAGES == 1)
  {
    cout<<"\nempty_image after decryption = ";
    printImageContents(img_dec);
  }
   
  if(DEBUG_IMAGES == 1)
  {
    cv::Mat img_reshape(m,n,CV_8UC3,dec_vec);
    cv::imwrite("airplane_rowcolswap_decrypted.png",img_reshape);
  }
  
  return 0;
}
