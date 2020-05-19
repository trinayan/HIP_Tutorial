#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{

  int n=1024000;

  //Input vectors
  double *h_a;
  double *h_b;

  //Output vectors
  double *h_c;

  // Size, in bytes, of each vector
  size_t bytes = n*sizeof(double);


  // Allocate memory for each vector on host
   h_a = (double*)malloc(bytes);
   h_b = (double*)malloc(bytes);
   h_c = (double*)malloc(bytes);

   int i;
    // Initialize vectors on host
   for( i = 0; i < n; i++ ) {
        h_a[i] = i;
        h_b[i] = i;
    }

    //Add the two vectors
    for( i = 0; i < n; i++ ) {
        h_c[i] = h_a[i] + h_b[i]; 
    }



    // Sum up vector c and print result divided by n, this should equal 1 within error
   
     // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
