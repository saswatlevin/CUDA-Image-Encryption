  #include <iostream> /*For IO*/
  #include <cstdint>  /*For standard variable types*/
  #include "functions.hpp"
  #include "kernel.hpp"
  
  using namespace std;
  using namespace cv;
  
  int main()
  { 
    
    uint16_t m=0,n=0,total=0;
    uint8_t number_cat_map_rounds=0,temp=0;
    std::string type=std::string("");
    
    cv::Mat image;
    cv::Mat fractal;
    
    /*LOAD AND SQUARE IMAGE. GET CATMAP ROUNDS*/
    image=cv::imread("airplane.png",cv::IMREAD_COLOR);
    //fractal=cv::imread("Gradient.png",cv::IMREAD_COLOR);
    
    mt19937 seeder1(time(0));
    uniform_int_distribution<int> intGen(CATMAP_ROUND_LOWER, CATMAP_ROUND_UPPER);
    auto rounds=intGen(seeder1); 
   
    if(RESIZE_TO_DEBUG==1)
    {
      cv::resize(image,image,cv::Size(100,100));
      //cv::resize(fractal,fractal,cv::Size(100,100));
    }
    
    getSquareImage(image,"original_dimensions.txt",RESIZE_TO_MAXIMUM);
    
    m=(uint16_t)image.rows;
    n=(uint16_t)image.cols;
    
    cout<<"\nm= "<<m<<"\tn= "<<n;
    type=type2str(image.type());
    cout<<"\nimage type= "<<type;
    total=(m*n);
    cout<<"\ntotal="<<total;  
    
   
    bool isNotDecimal=0;

    
    
    /*Declarations*/
    
    /*CPU vector declarations and allocations*/
    double *P1=(double*)malloc(sizeof(double)*total);
    double *P2=(double*)malloc(sizeof(double)*total);
    uint8_t *img_vec=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
    uint8_t *img_empty=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
    uint16_t *U=(uint16_t*)malloc(sizeof(uint16_t)*m);
    uint16_t *V=(uint16_t*)malloc(sizeof(uint16_t)*m);
    uint8_t *fractal_vec=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
    uint8_t *original_img_vec=(uint8_t*)malloc(sizeof(uint8_t)*total*3);
    
    
    for(uint32_t i=0;i<m;++i)
    {
      U[i]=0;
      V[i]=0;
    }    
    
    for(uint32_t i=0;i<total*3;++i)
    {
      img_vec[i]=0;
      img_empty[i]=0;
      fractal_vec[i]=0;
    } 
    
    for(uint32_t i=0;i<total;++i)
    {
      P1[i]=0;
      P2[i]=0;
    }
    cout<<"\nFInished initializing P1,P2,img_vec,fractal_vec,U,V";
    /*GPU vector declarations*/
    uint8_t *gpuimgIn;
    uint8_t *gpuimgOut;
    uint16_t *gpuU;
    uint16_t *gpuV;
    uint8_t *gpuFrac;
     
    /*FLATTEN IMAGE*/
    flattenImage(image,img_vec); 
   
   /*GENERATE RELOCATION VECTORS*/
   genRelocVecEnc(U,P1,m,n,"constants1.txt");
   genRelocVecEnc(V,P2,n,m,"constants2.txt");
   cout<<"\nGenerated Relocation Vectors"; 
    
    /*Checking P1,P2,U and V*/
    if (DEBUG_VECTORS==1)
    {
      cout<<"\nP1=";
      printFloatVector(P1);
    
      cout<<"\nU=";
      for(uint32_t i=0;i<m;++i)
      {
        printf("%d ",U[i]);
      }
    
      cout<<"\nP2=";
      printFloatVector(P2);
      cout<<"\nV=";
    
      for(uint32_t i=0;i<m;++i)
      {
        printf("%d ",V[i]);
      }
      
      cout<<"\n";
  }
   
    
    
   if(PRINT_IMAGES==1)
   {
     printImageContents(image);
   }
    
   if(DEBUG_VECTORS==1) 
   {
     
     
     cout<<"\nimg_vec before Enc_GenCatMap=";
     for(uint32_t i=0;i<total*3;++i)
     {
        
        printf("%d ",img_vec[i]);
     }
     

     std::ofstream file("img_vec.txt");
     std::string image_elements=std::string("");
     if(!file)
     {
       cout<<"Could not create img_vec.txt\nExiting...";
       exit(1);
     }
     
     for(uint32_t i=0;i<total*3;++i)
     {
       image_elements.append(std::to_string(img_vec[i]));
       image_elements.append("\n");
     }
     
     file<<image_elements;
     file.close();
   
  }
   
   
   
    /*WARMUP
    dim3 grid_warm_up(1,1,1);
    dim3 block_warm_up(1,1,1);
    
    run_WarmUp(grid_warm_up,block_warm_up);*/
    
    /*ARNOLD IMAGE MAPPING*/
    
    cudaMallocManaged((void**)&gpuimgIn,total*3*sizeof(uint8_t));
    cudaMallocManaged((void**)&gpuimgOut,total*3*sizeof(uint8_t)); 
    
    for (uint32_t i=0;i<total*3;++i)
    {
       gpuimgIn[i]=img_vec[i];
       gpuimgOut[i]=0;
    }
    
    /*dim3 grid_ar_map_img(m,n,1);
    dim3 block_ar_map_img(3,1,1);
    
    cout<<"\nBefore Armapimg kernel call"; 
    
    for(uint32_t i=0;i<max((uint16_t)rounds,5);++i)
    { cout<<"\nIn Armapimg kernel call loop";
      run_ArMapImg(gpuimgIn,gpuimgOut,grid_ar_map_img,block_ar_map_img);
      for(uint32_t i=0;i<total*3;++i)
      {
        temp=gpuimgIn[i];
        gpuimgIn[i]=gpuimgOut[i];
        gpuimgOut[i]=temp;
      }
    }
    
    for(uint32_t i=0;i<total*3;++i)
    {
      img_vec[i]=gpuimgOut[i];
    } 
   
    if(DEBUG_VECTORS==1)
    {
      cout<<"\nimgvec after ArMapImg and Shuffle=";
      for(uint32_t i=0;i<total*3;++i)
      {
         printf("%d ",img_vec[i]);
      }
    }*/
    
   /*FRACTAL XORING*/
  flattenImage(fractal,fractal_vec);

  cudaMallocManaged((void**)&gpuFrac,total*3*sizeof(uint8_t));
  
  for(int i=0;i<total*3;++i)
  {
    gpuimgIn[i]=img_vec[i];
  }  
  
  
  for(int i=0;i<total*3;++i)
  {
    gpuFrac[i]=fractal_vec[i];
  }
  
  dim3 grid_frac_xor(m*n,1,1);
  dim3 block_frac_xor(3,1,1);
  
  run_FracXor(gpuimgIn,gpuimgOut,gpuFrac,grid_frac_xor,block_frac_xor);
  
  
  for(int i=0;i<total*3;++i)
  {
    temp=gpuimgIn[i];
    gpuimgIn[i]=gpuimgOut[i];
    gpuimgOut[i]=temp;
  }  

  for(int i=0;i<total*3;++i)
  {
    img_vec[i]=gpuimgOut[i];
  }
   
   
   
   /*ARNOLD MAP ENCRYPTION*/
   cudaMallocManaged((void**)&gpuU,m*sizeof(uint16_t));
   cudaMallocManaged((void**)&gpuV,m*sizeof(uint16_t));
   
   for(uint32_t i=0;i<m;++i)
   {
     gpuU[i]=U[i];
     gpuV[i]=V[i];
   }

   dim3 grid_enc_gen_cat_map(m,n,1);
   dim3 block_enc_gen_cat_map(3,1,1);
   
   
   
  for(uint32_t i=0;i<PERM_ROUNDS;++i)
  {
   run_EncGenCatMap(gpuimgIn,gpuimgOut,gpuU,gpuV,grid_enc_gen_cat_map,block_enc_gen_cat_map);
   for(uint32_t i=0;i<total*3;++i)
   {
     temp=gpuimgIn[i];
     gpuimgIn[i]=gpuimgOut[i];
     gpuimgOut[i]=temp;
   }
  }
   
   for(uint32_t i=0;i<total*3;++i)
   {
     img_vec[i]=gpuimgOut[i]; 
   }
   
   if(DEBUG_VECTORS==1)
   {
     cout<<"\nimg_vec after Enc_GenCatMap=";
     for(uint32_t i=0;i<total*3;++i)
     {
       printf("%d ",img_vec[i]);
     }
   }
   
   /*Converting img_reshape to Mat image*/
   if(DEBUG_IMAGES==1)
   {
     cv::Mat img_reshape(m,n,CV_8UC3,img_vec);
     cv::imwrite("airplane_encrypted.png",img_reshape);
   }
   return 0; 
  }
  

