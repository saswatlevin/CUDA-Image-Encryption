#ifndef CONFIG_H /*These two lines are to ensure no errors if the header file is included twice*/
#define CONFIG_H

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  //define something for Windows (32-bit and 64-bit, this part is common)
 std::string separator;
  #ifdef _WIN64
    //define something for Windows (64-bit only)
    separator = "\\";
  #else
    //define something for Windows (32-bit only)
    separator = "\\";
  #endif

#elif __linux__
  std::string separator = "/";
  
 
#elif __unix__ // all unices not caught above
  // Unix
   std::string separator = "/";
  
#elif defined(_POSIX_VERSION)
  // POSIX
  std::string separator = "/";

#else
  #error "Unknown compiler"
#endif

#define RESIZE_TO_DEBUG                1
#define DEBUG_VECTORS                  0
#define DEBUG_IMAGES                   1
#define DEBUG_INTERMEDIATE_IMAGES      1

#define PRINT_TIMING                   0
#define PRINT_IMAGES                   0

#define LOWER_LIMIT                    0.000001
#define UPPER_LIMIT                    0.09

#define NUMBER_OF_BITS                 31
#define INIT                           100

#define ROW_COL_ROTATION               1
#define DIFFUSION                      1
#define ROW_COL_SWAPPING               1 
#define BIT_RETURN(A,LOC) (( (A >> LOC ) & 0x1) ? 1:0)

namespace config
{
  uint32_t rows = 235;
  uint32_t cols = 235;
  int read_mode = 24;
  int write_mode = 0;
  int lower_limit = 1;
  int upper_limit = (rows * cols * 4) + 1;
  int seed_lut_gen_1 = 1000;
  int seed_lut_gen_2 = 2000;

  int seed_row_rotate = 1000000;
  int seed_col_rotate = 2000000;
  int seed_diffusion = 3000000;
  
  

  
  typedef struct
  {
    double x_init;
    double y_init;
    double alpha;
    double beta;
  }slmm; 

  slmm slmm_map;
  
  std::string image_name = "mountain_alpha";
  std::string encrypted_image = image_name + "_encrypted_";
  std::string decrypted_image = image_name + "_decrypted_";
  std::string swapped_image = image_name + "_swapped";
  std::string unswapped_image =  image_name + "_unswapped";
  std::string rotated_image = image_name + "_rotated";
  std::string unrotated_image =  image_name + "_unrotated";
  std::string diffused_image = image_name + "_diffused";
  std::string undiffused_image = image_name + "_undiffused";
  std::string extension = ".png";
  std::string input = "input"; 
  
  std::string input_image_path = input + separator + image_name + extension;
  std::string swapped_image_path = swapped_image + extension;
  std::string unswapped_image_path = unswapped_image + extension;
  std::string rotated_image_path = rotated_image + extension;
  std::string unrotated_image_path = unrotated_image + extension;
  std::string diffused_image_path = diffused_image + extension;
  std::string undiffused_image_path = undiffused_image + extension; 
  std::string encrypted_image_path = encrypted_image + extension;
  std::string decrypted_image_path = decrypted_image + extension;   
}  

#endif