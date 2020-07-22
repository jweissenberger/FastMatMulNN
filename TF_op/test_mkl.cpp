#include "omp.h"
#include "mkl.h"
#include <stdio.h>
#include <iostream>

#define SIZE 5000

int main(int args, char *argv[]){

	double *a, *b, *c, *d;
	a = new double [SIZE*SIZE];
	b = new double [SIZE*SIZE];
	c = new double [SIZE*SIZE];
	d = new double [SIZE*SIZE];

	double alpha=1, beta=1;
	int m=SIZE, n=SIZE, k=SIZE, lda=SIZE, ldb=SIZE, ldc=SIZE, i=0, j=0;
	char transa='n', transb='n';

	omp_set_num_threads(12);
	std::cout << mkl_get_max_threads() << "," << mkl_get_dynamic() << "," << omp_get_num_threads() << std::endl;

	for( i=0; i<SIZE; i++){
		for( j=0; j<SIZE; j++){
			a[i*SIZE+j]= (double)(i+j);
			b[i*SIZE+j]= (double)(i*j);
			c[i*SIZE+j]= (double)0;
			d[i*SIZE+j]= (double)0;
		}
	}

	double start_time = omp_get_wtime();
#pragma omp parallel num_threads(2)
	{
#pragma omp single 
		{
				std::cout << "omp threads: " << omp_get_num_threads() << ", mkl max threads: " << mkl_get_max_threads() << std::endl;
#pragma omp task 
			{
				std::cout << "working on 1st one with this many threads:" << mkl_get_max_threads() << std::endl;
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
			}
			/*printf("row a c ");
			  for ( i=0;i<10;i++){
			  printf("%d: %f %f ", i, a[i*SIZE], c[i*SIZE]);
			  }*/

			//omp_set_num_threads(1);

			/*for( i=0; i<SIZE; i++){
			  for( j=0; j<SIZE; j++){
			  a[i*SIZE+j]= (double)(i+j);
			  b[i*SIZE+j]= (double)(i*j);
			  c[i*SIZE+j]= (double)0;
			  }
			  }*/
//#pragma omp task 
			{
				std::cout << "working on 2nd one..." << std::endl;
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
			}
#pragma omp taskwait
		}
		} // end of parallel section
	double elapsed_time = omp_get_wtime() - start_time;
	/*printf("row a c ");
	  for ( i=0;i<10;i++){
	  printf("%d: %f %f ", i, a[i*SIZE],
	  c[i*SIZE]);
	  }*/

	std::cout << mkl_get_max_threads() << "," << mkl_get_dynamic() << ", elapsed time was " << elapsed_time << std::endl;
//	omp_set_num_threads(12);

	for( i=0; i<SIZE; i++){
		for( j=0; j<SIZE; j++){
			a[i*SIZE+j]= (double)(i+j);
			b[i*SIZE+j]= (double)(i*j);
			c[i*SIZE+j]= (double)0;
		}
	}
	start_time = omp_get_wtime();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
	elapsed_time = omp_get_wtime() - start_time;
	std::cout << mkl_get_max_threads() << ", elapsed time was " << elapsed_time << std::endl;
	/*printf("row a c ");
	for ( i=0;i<10;i++){
		printf("%d: %f %f ", i, a[i*SIZE],
				c[i*SIZE]);
	}*/

	delete [] a;
	delete [] b;
	delete [] c;
}
