#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include "gpuerrors.h"

#define D 128
#define D_L 100
#define N_ref 1000000
#define m 50
#define T1 25
#define T2 32
#define T3 128
#define T4 1000
#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// ===========================> Functions Prototype <===============================
int fvecs_read (const char *fname, int d, int n, float *a);
int ivecs_write (const char *fname, int d, int n, const int *v);
void get_inputs(int argc, char *argv[], unsigned int& N, unsigned int& K);
void gpuKernels(float* ref, float* query, int* hist, unsigned int N, unsigned int K, double* gpu_kernel_time);
__global__ void DistanceCal(float* result,float* x,float* q);
__global__ void MinMax(float* result,float* min,float* max);
__global__ void Histogram(float* result,float* min,float* max,unsigned int K,int* hist_gpu);
__global__ void put_zero(int* hist_gpu);
__global__ void Histogram2(float* result,float* min,float* max,unsigned int K,int* result2) ;
__global__ void reduce(int* result2,unsigned int k,int* hist_gpu) ;
__global__ void put_zero2(int* result2,unsigned int K) ;
__global__ void MinMax1(float* result,float* min1,float* max1) ;
__global__ void MinMax2(float* min,float* max,float* min1,float* max1) ;
__global__ void Histogram3(float* result,float* min,float* max,unsigned int K,int* hist_gpu) ;
__global__ void Histogram4(float* result,float* min,float* max,unsigned int K,int* hist_gpu) ;
__global__ void Histogram5(float* result,float* min,float* max,unsigned int K,int* hist_gpu) ;
__global__ void Histogram6(float* result,float* min,float* max,unsigned int K,int* hist_gpu) ;
// =================================================================================

int main(int argc, char *argv[]) {

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);

    // get parameters from command line
    unsigned int N, K;
    get_inputs(argc, argv, N, K);

    // allocate memory in CPU for calculation
    float* reference; // reference vectors
    float* query; // query points
    int* hist;

    // Memory Allocation
	reference = (float*)malloc(1000000 * 128 * sizeof(float));
	query = (float*)malloc(N * 128 * sizeof(float));
	hist = (int*)malloc(N * K * sizeof(int));
	

    // fill references, query and labels with the values read from files
    fvecs_read("/home/data/ref.fvecs", D, N_ref, reference);
    fvecs_read("/home/data/query.fvecs", D, N, query);
    
    // time measurement for GPU calculation
    double gpu_kernel_time = 0.0;
    clock_t t0 = clock();
	  gpuKernels(reference, query, hist, N, K, &gpu_kernel_time);
    clock_t t1 = clock();

    printf("k=%d n=%d GPU=%g ms GPU-Kernels=%g ms\n",
    K, N, (t1-t0)/1000.0, gpu_kernel_time);

    // write the output to a file
    ivecs_write("outputs.ivecs", K, N, hist);
	
	/*for (int f = 0;f < N*K;f++){
		printf("%d\n",hist[f]);
	}*/
    
    // free allocated memory for later use
    free(reference);
    free(hist);
    free(query);

    return 0;
}

//-----------------------------------------------------------------------------
void gpuKernels(float* reference, float* query, int* hist, unsigned int N, unsigned int K, double* gpu_kernel_time) {

   // Memory Allocation and Copy to Device
	float* x;
	float* q;
	float* result ;
	int* result2 ;
	float* min;
	float* max;
	float* min1;
	float* max1;
	int* hist_gpu;
	HANDLE_ERROR(cudaMalloc((void**)&x, 128 * 1000000 * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&q, m * 128 * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&result, m * 1000000 * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(x, reference,128 * 1000000 * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&min, m * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&max, m * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&hist_gpu, m * K * sizeof(int)));
	//HANDLE_ERROR(cudaMalloc((void**)&result2, m * K *1024 * sizeof(int)));
	//HANDLE_ERROR(cudaMalloc((void**)&min1, m * 1000 * sizeof(float)));
	//HANDLE_ERROR(cudaMalloc((void**)&max1, m * 1000 * sizeof(float)));
	dim3 Dimgrid1(1000000/T2,m/T1,1) ;
	dim3 Dimblock1(T2,T1,1) ;
	dim3 Dimgrid2(m,1000000 / T4,1) ;
	dim3 Dimblock2(T4,1,1) ;
	dim3 Dimgrid3(m,1000,1) ;
	dim3 Dimblock3(1000,1,1) ;
	dim3 Dimgrid4(m,K,1) ;
	dim3 Dimblock4(512,1,1) ;
	dim3 Dimgrid5(m,K,1) ;
	dim3 Dimblock5(24,1,1) ;
	dim3 Dimgrid6(m,1000,1) ;
	dim3 Dimblock6(512,1,1) ;
	//put_zero2<<<Dimgrid5,Dimblock5>>>(result2,K) ;
	

	GpuTimer timer;
    timer.Start();
    
    //Put Your Main Code for Computation
	if (K < 257){
	for (int i = 0; i < N / m;i++)
	{
		put_zero<<<K,m>>>(hist_gpu) ;
		HANDLE_ERROR(cudaMemcpy(q, &query[i * m * 128],128 * m * sizeof(float), cudaMemcpyHostToDevice));
		DistanceCal<<<Dimgrid1,Dimblock1>>>(result,x,q);
		MinMax<<< m , 1000 >>>(result,min,max);
		//MinMax1<<<Dimgrid6,Dimblock6>>>(result,min1,max1);
		//MinMax2<<<m,512>>>(min,max,min1,max1);
		//Histogram<<<Dimgrid2,Dimblock2>>>(result,min,max,K,hist_gpu);
		//Histogram2<<<Dimgrid3,Dimblock3>>>(result,min,max,K,result2) ;
		//reduce<<<Dimgrid4,Dimblock4>>>(result2,K,hist_gpu) ;
		Histogram3<<<m,1024>>>(result,min,max,K,hist_gpu);
		HANDLE_ERROR(cudaMemcpy(&hist[i * m * K],hist_gpu, m * K * sizeof(float), cudaMemcpyDeviceToHost));
	}
	}
	else if ((K > 256)&&(K < 513)){
	for (int i = 0; i < N / m;i++)
	{
		put_zero<<<K,m>>>(hist_gpu) ;
		HANDLE_ERROR(cudaMemcpy(q, &query[i * m * 128],128 * m * sizeof(float), cudaMemcpyHostToDevice));
		DistanceCal<<<Dimgrid1,Dimblock1>>>(result,x,q);
		MinMax<<< m , 1000 >>>(result,min,max);
		//MinMax1<<<Dimgrid6,Dimblock6>>>(result,min1,max1);
		//MinMax2<<<m,512>>>(min,max,min1,max1);
		//Histogram<<<Dimgrid2,Dimblock2>>>(result,min,max,K,hist_gpu);
		//Histogram2<<<Dimgrid3,Dimblock3>>>(result,min,max,K,result2) ;
		//reduce<<<Dimgrid4,Dimblock4>>>(result2,K,hist_gpu) ;
		Histogram4<<<m,1024>>>(result,min,max,K,hist_gpu);
		HANDLE_ERROR(cudaMemcpy(&hist[i * m * K],hist_gpu, m * K * sizeof(float), cudaMemcpyDeviceToHost));
	}
	}
	else if ((K > 512)&&(K < 1025)){
	for (int i = 0; i < N / m;i++)
	{
		put_zero<<<K,m>>>(hist_gpu) ;
		HANDLE_ERROR(cudaMemcpy(q, &query[i * m * 128],128 * m * sizeof(float), cudaMemcpyHostToDevice));
		DistanceCal<<<Dimgrid1,Dimblock1>>>(result,x,q);
		MinMax<<< m , 1000 >>>(result,min,max);
		//MinMax1<<<Dimgrid6,Dimblock6>>>(result,min1,max1);
		//MinMax2<<<m,512>>>(min,max,min1,max1);
		//Histogram<<<Dimgrid2,Dimblock2>>>(result,min,max,K,hist_gpu);
		//Histogram2<<<Dimgrid3,Dimblock3>>>(result,min,max,K,result2) ;
		//reduce<<<Dimgrid4,Dimblock4>>>(result2,K,hist_gpu) ;
		Histogram5<<<m,1024>>>(result,min,max,K,hist_gpu);
		HANDLE_ERROR(cudaMemcpy(&hist[i * m * K],hist_gpu, m * K * sizeof(float), cudaMemcpyDeviceToHost));
	}
	}
	else{
	for (int i = 0; i < N / m;i++)
	{
		put_zero<<<K,m>>>(hist_gpu) ;
		HANDLE_ERROR(cudaMemcpy(q, &query[i * m * 128],128 * m * sizeof(float), cudaMemcpyHostToDevice));
		DistanceCal<<<Dimgrid1,Dimblock1>>>(result,x,q);
		MinMax<<< m , 1000 >>>(result,min,max);
		//MinMax1<<<Dimgrid6,Dimblock6>>>(result,min1,max1);
		//MinMax2<<<m,512>>>(min,max,min1,max1);
		//Histogram<<<Dimgrid2,Dimblock2>>>(result,min,max,K,hist_gpu);
		//Histogram2<<<Dimgrid3,Dimblock3>>>(result,min,max,K,result2) ;
		//reduce<<<Dimgrid4,Dimblock4>>>(result2,K,hist_gpu) ;
		Histogram6<<<m,1024>>>(result,min,max,K,hist_gpu);
		HANDLE_ERROR(cudaMemcpy(&hist[i * m * K],hist_gpu, m * K * sizeof(float), cudaMemcpyDeviceToHost));
	}
	}
	
    
  	timer.Stop();
	  *gpu_kernel_time = timer.Elapsed();

    //Copy to Host and Free the Memory
	HANDLE_ERROR(cudaFree(x));
	HANDLE_ERROR(cudaFree(q));
	HANDLE_ERROR(cudaFree(result));
	HANDLE_ERROR(cudaFree(min));
	HANDLE_ERROR(cudaFree(max));
	HANDLE_ERROR(cudaFree(hist_gpu));

}
//-----------------------------------------------------------------------------
void get_inputs(int argc, char *argv[], unsigned int& N, unsigned int& K)
{
    if (
	argc != 3 ||
	atoi(argv[1]) < 0 || atoi(argv[1]) > 10000 ||
	atoi(argv[2]) < 0 || atoi(argv[2]) > 5000
	) {
        printf("<< Error >>\n");
        printf("Enter the following command:\n");
        printf("\t./nn  N  K\n");
        printf("\t\tN must be between 0 and 10000\n");
        printf("\t\tK must be between 0 and 5000\n");
		exit(-1);
    }
	N = atoi(argv[1]);
	K = atoi(argv[2]);
}
//-----------------------------------------------------------------------------
int fvecs_read (const char *fname, int d, int n, float *a)
{
  FILE *f = fopen (fname, "r");
  if (!f) {
    fprintf (stderr, "fvecs_read: could not open %s\n", fname);
    perror ("");
    return -1;
  }

  long i;
  for (i = 0; i < n; i++) {
    int new_d;

    if (fread (&new_d, sizeof (int), 1, f) != 1) {
      if (feof (f))
        break;
      else {
        perror ("fvecs_read error 1");
        fclose(f);
        return -1;
      }
    }

    if (new_d != d) {
      fprintf (stderr, "fvecs_read error 2: unexpected vector dimension\n");
      fclose(f);
      return -1;
    }

    if (fread (a + d * (long) i, sizeof (float), d, f) != d) {
      fprintf (stderr, "fvecs_read error 3\n");
      fclose(f);
      return -1;
    }
  }
  fclose (f);

  return i;
}


int ivecs_write (const char *fname, int d, int n, const int *v)
{
  FILE *f = fopen (fname, "w");
  if (!f) {
    perror ("ivecs_write");
    return -1;
  }

  int i;
  for (i = 0 ; i < n ; i++) {
    fwrite (&d, sizeof (d), 1, f);
    fwrite (v, sizeof (*v), d, f);
    v+=d;
  }
  fclose (f);
  return n;
}



__global__ void DistanceCal(float* result,float* x,float* q)
{
	int i = by * T1 + ty;
	int j = bx * T2 + tx;
	int r ;
	float temp;
	
	__shared__ float q_sh[T1][T3];
	__shared__ float x_sh[T3][T2];
	float t=0 ;
	for (int k = 0 ; k < 128 / T3 ; k++){
		for(r=0;r<T3/T2;r++)
		{
			q_sh[ty][r*T2+tx]=q[(i*128)+k*T3+r*T2+tx];
		}
		for(r=ty;r<T3;r+=T1)
		{
			x_sh[r][tx]=x[j*128+k*T3+r] ;
		}
		/*for(r=0;r<T3/T1;r++)
		{
			x_sh[tx][r*T1+ty] = x[j*128+k*T3+r*T1+ty] ;
		}*/
		__syncthreads() ;
		for(r=0;r<T3;r++)
		{
			temp = q_sh[ty][r]-x_sh[r][tx];
			t+= temp * temp ;
		}
		__syncthreads() ;
		
	}
	result[i*1000000+j]=sqrt(t) ;

}
/*
__global__ void MinMax(float* result,float* min,float* max)
{
	float temp1=result[bx*1000000],temp2=result[bx*1000000],temp3;
	for(int i=1 ;i<1000000;i++)
	{
		temp3=result[bx*1000000+i] ;
		if(temp3<temp1)
		temp1=temp3 ;
		if(temp3>temp2)
		temp2=temp3 ;
	}	
	min[bx]=temp1 ;
	max[bx]=temp2 ;
}*/

__global__ void MinMax(float* result,float* min,float* max)
{
	__shared__ float min_sh[1000] ;
	__shared__ float max_sh[1000] ;
	float temp1=result[bx*1000000+tx*1000],temp2,temp3,temp4;
	temp2=temp1 ;
	for(int i=1 ;i<1000;i++)
	{
		temp3=result[bx*1000000+tx*1000+i] ;
		if(temp3<temp1)
		temp1=temp3 ;
		if(temp3>temp2)
		temp2=temp3 ;
	}	
	
	min_sh[tx]=temp1 ;
	max_sh[tx]=temp2 ;
	__syncthreads() ;
	if(tx==0)
	{
		temp1=min_sh[0] ;
		for(int i=1 ;i<1000;i++)
	{
		temp3=min_sh[i] ;
		if(temp3<temp1)
		temp1=temp3 ;
	}
		min[bx]=temp1 ;
	}	
	if(tx==1)
	{
		temp2=max_sh[0] ;
		for(int i=1 ;i<1000;i++)
	{
		temp4=max_sh[i] ;
		if(temp4>temp2)
		temp2=temp4 ;
	}
		max[bx]=temp2 ;	
	}	
}


__global__ void MinMax1(float* result,float* min1,float* max1)
{
	__shared__ float min_sh1[512] ;
	__shared__ float max_sh1[512] ;
	__shared__ float min_sh2[512] ;
	__shared__ float max_sh2[512] ;
	float temp1,temp2,j,flag=0 ;
	temp1 = result[bx*1000000+by*1000+2*tx] ;
	temp2 = result[bx*1000000+by*1000+2*tx+1] ;
	if(temp1<temp2)
	{
		min_sh1[tx]=temp1 ;
		max_sh1[tx]=temp2 ;
	}
	else
	{
		min_sh1[tx]=temp2 ;
		max_sh1[tx]=temp1 ;
	}
	for(j=1;j<512;j++)
	{
		__syncthreads() ;
		if(tx<(256/j))
			{
				if(flag==0)
				{
					if(min_sh1[2*tx]<min_sh1[2*tx+1])
						min_sh2[tx]=min_sh1[2*tx] ;
					else
						min_sh2[tx]=min_sh1[2*tx+1] ;
						
					if(max_sh1[2*tx]>max_sh1[2*tx+1])
						max_sh2[tx]=max_sh1[2*tx] ;
					else
						max_sh2[tx]=max_sh1[2*tx+1] ;
				}
				else
				{
					if(min_sh2[2*tx]<min_sh2[2*tx+1])
						min_sh1[tx]=min_sh2[2*tx] ;
					else
						min_sh1[tx]=min_sh2[2*tx+1] ;
						
					if(max_sh2[2*tx]>max_sh2[2*tx+1])
						max_sh1[tx]=max_sh2[2*tx] ;
					else
						max_sh1[tx]=max_sh2[2*tx+1] ;
				}
			
				flag=1-flag ;
			}
	}
	__syncthreads() ;
	if(tx==0)
	{
		min1[bx*1000+by] = min_sh2[0] ;
		max1[bx*1000+by] = max_sh2[0] ;	
	}
	
}




__global__ void MinMax2(float* min,float* max,float* min1,float* max1)
{
	__shared__ float min_sh1[512] ;
	__shared__ float max_sh1[512] ;
	__shared__ float min_sh2[512] ;
	__shared__ float max_sh2[512] ;
	float temp1,temp2,temp3,temp4,j,flag = 0 ;
	temp1 = min1[bx*1000+((2*tx) % 1000) ] ;
	temp2 = min1[bx*1000+((2*tx+1) % 1000)] ;
	temp3 = max1[bx*1000+((2*tx) % 1000) ] ;
	temp4 = max1[bx*1000+((2*tx+1) % 1000)] ;
	if(temp1<temp2)
	{
		min_sh1[tx]=temp1 ;
	}
	else
	{
		min_sh1[tx]=temp2 ;
	}
	
	if(temp3>temp4)
	{
		max_sh1[tx]=temp3 ;
	}
	else
	{
		max_sh1[tx]=temp4 ;
	}
	
	for(j=1;j<512;j++)
	{
		__syncthreads() ;
		if(tx<(256/j))
			{
				if(flag==0)
				{
					if(min_sh1[2*tx]<min_sh1[2*tx+1])
						min_sh2[tx]=min_sh1[2*tx] ;
					else
						min_sh2[tx]=min_sh1[2*tx+1] ;
						
					if(max_sh1[2*tx]>max_sh1[2*tx+1])
						max_sh2[tx]=max_sh1[2*tx] ;
					else
						max_sh2[tx]=max_sh1[2*tx+1] ;
				}
				else
				{
					if(min_sh2[2*tx]<min_sh2[2*tx+1])
						min_sh1[tx]=min_sh2[2*tx] ;
					else
						min_sh1[tx]=min_sh2[2*tx+1] ;
						
					if(max_sh2[2*tx]>max_sh2[2*tx+1])
						max_sh1[tx]=max_sh2[2*tx] ;
					else
						max_sh1[tx]=max_sh2[2*tx+1] ;
				}
			
				flag=1-flag ;
			}
	}
	__syncthreads() ;
	if(tx==0)
	{
		min[bx] = min_sh2[0] ;
		max[bx] = max_sh2[0] ;	
	}
	
}



__global__ void Histogram(float* result,float* min,float* max,unsigned int K,int* hist_gpu){
	
	float data = result[bx * 1000000 + by * T4 + tx];
	float min2 = min[bx];
	float max2=max[bx] ;
	int r ;
	if(data==max2) 
	r=K-1 ;
	else
	r = floor(((data - min2)/(max2 - min2)) * K);
	atomicAdd(&hist_gpu[bx * K + r],1);
}
/*
__global__ void Histogram2(float* result,float* min,float* max,unsigned int K,int* result2){
	__shared__ int bins[1024] ;
	int b = tx<1000 ;
	int r=0,i,j ;
	if(b)
	{
	float data = result[bx * 1000000 + by * 1000 + tx];
	float min2 = min[bx] ;
	float max2=max[bx] ;
	if(data==max2) 
	r=K-1 ;
	else
	r = floor(((data - min2)/(max2 - min2)) * K);
	}
	for(i=0;i<K;i++)
	{
		bins[tx]=(r==i) && b ;
		
		for(j=1;j<1024;j*=2)
		{
			__syncthreads() ;
			if(tx<(512/j))
			{
				bins[2*tx*j]=bins[2*tx*j]+bins[2*tx*j+j] ;
			}
		}
		if(tx==0)
		{
			result2[bx*K*1024+i*1024+by]=bins[0] ;
		}
		__syncthreads() ;
	}
}
*/


__global__ void Histogram2(float* result,float* min,float* max,unsigned int K,int* result2){
	__shared__ int bins1[1024] ;
	__shared__ int bins2[1024] ;
	int r=0,i,j,flag ;
	float data = result[bx * 1000000 + by * 1000 + tx];
	float min2 = min[bx] ;
	float max2 = max[bx] ;
	if(data==max2) 
	r=K-1 ;
	else
	r = floor(((data - min2)/(max2 - min2)) * K);
	//if(tx<24)
	//bins1[tx+1000]=0 ;
	for(i=0;i<K;i++)
	{
		bins1[tx] = (r==i) ;
		flag=0 ;
		for(j=1;j<1024;j*=2)
		{
			__syncthreads() ;
			if(tx<(512/j))
			{
				if(flag==0)
				bins2[tx]=bins1[2*tx]+bins1[2*tx+1] ;
				else
				bins1[tx]=bins2[2*tx]+bins2[2*tx+1] ;
				flag=1-flag ;
			}
		}
		__syncthreads() ;
		if(tx==0)
		{
			result2[bx*K*1024+i*1024+by]=bins1[0] ;
		}
		
	}
}


__global__ void reduce(int* result2,unsigned int K,int* hist_gpu)
{
	__shared__ int result_sh1[1024] ;
	__shared__ int result_sh2[1024] ;
	int j,flag=0;
	result_sh1[2*tx] = result2[bx*K*1024+by*1024+2*tx] ;
	result_sh1[2*tx+1] = result2[bx*K*1024+by*1024+2*tx+1] ;
	for(j=1;j<1024;j*=2)
		{
			__syncthreads() ;
			if(tx<(512/j))
			{
				if(flag==0)
				result_sh2[tx]=result_sh1[2*tx]+result_sh1[2*tx+1] ;
				else
				result_sh1[tx]=result_sh2[2*tx]+result_sh2[2*tx+1] ;
				flag=1-flag ;
			}
		}
		__syncthreads() ;
		if(tx==0)
		{
			hist_gpu[bx*K+by] = result_sh1[0] ;
		}
}

/*
__global__ void Histogram3(float* result,float* min,float* max,unsigned int K,int* hist_gpu)
{
	__shared__ int bins1[256][8] ;
	__shared__ int bins2[256][8] ;
	float min2 = min[bx] ;
	float max2 = max[bx] ;
 	int i,first_index,last_index,r,position,flag=0,my_K,b;
 	float data ;
 	if(tx<K)
 	{
 	for(i=0;i<8;i++)
	{
		bins1[tx][i]=0 ;
	}
	}
	if(tx<576)
	{
		first_index = tx * 977 ;
		last_index = first_index + 977 ;
	}
	else
	{
		first_index = 576 * 977 + (tx-576) * 976 ;
		last_index = first_index + 976 ;
	}
	//last_index = first_index + 976 + (tx<576) ;
	position = first_index/125000 ;
	for(i=first_index ; i<last_index;i++)
	{
	data = result[bx * 1000000 + i];
	if(data==max2) 
	r=K-1 ;
	else
	r = floor(((data - min2)/(max2 - min2)) * K);
	atomicAdd(&bins1[r][position],1);
	}
	__syncthreads() ;
	my_K = (tx / 4) ;
	b = my_K < K ;
	for(i=1;i<8;i*=2)
	{
		if(((tx % 4) < (4/i)) && b)
		{
			if(flag==0)
			bins2[my_K][tx % 4]=bins1[my_K][2*(tx % 4)]+bins1[my_K][2*(tx % 4) + 1] ;
			else
			bins1[my_K][tx % 4]=bins2[my_K][2*(tx % 4)]+bins2[my_K][2*(tx % 4) + 1] ;
			flag=1-flag ;
		}
		__syncthreads() ;
	}
	if(((tx % 4)==0) && b)
	{
		hist_gpu[bx * K + my_K] = bins2[my_K][0] ;
	}
}
*/

__global__ void Histogram3(float* result,float* min,float* max,unsigned int K,int* hist_gpu)
{
	__shared__ int bins1[256][8] ;
	__shared__ int bins2[256][8] ;
	float min2 = min[bx] ;
	float max2 = max[bx] ;
 	int i,first_index,last_index,r,position,flag=0,my_K,b;
 	float data ;
 	if(tx<K)
 	{
 	for(i=0;i<8;i++)
	{
		bins1[tx][i]=0 ;
	}
	}
	if(tx<576)
	{
		first_index = tx * 977 ;
		last_index = first_index + 977 ;
	}
	else
	{
		first_index = 576 * 977 + (tx-576) * 976 ;
		last_index = first_index + 976 ;
	}
	position = first_index/125000 ;
	for(i=first_index ; i<last_index;i++)
	{
	data = result[bx * 1000000 + i];
	if(data==max2) 
	r=K-1 ;
	else
	r = floor(((data - min2)/(max2 - min2)) * K);
	atomicAdd(&bins1[r][position],1);
	}
	__syncthreads() ;
	my_K = (tx / 4) ;
	b = my_K < K ;
	if(b)
	bins2[my_K][tx % 4]=bins1[my_K][2*(tx % 4)]+bins1[my_K][2*(tx % 4) + 1] ;
	__syncthreads() ;
	if(((tx % 4) < 2) && b)
	bins1[my_K][tx % 4]=bins2[my_K][2*(tx % 4)]+bins2[my_K][2*(tx % 4) + 1] ;
	__syncthreads() ;
	if(((tx % 4)==0) && b)
	bins2[my_K][tx % 4]=bins1[my_K][2*(tx % 4)]+bins1[my_K][2*(tx % 4) + 1] ;
	__syncthreads() ;
	if(((tx % 4)==0) && b)
	{
		hist_gpu[bx * K + my_K] = bins2[my_K][0] ;
	}
}




__global__ void Histogram4(float* result,float* min,float* max,unsigned int K,int* hist_gpu)
{
	__shared__ int bins1[512][4] ;
	__shared__ int bins2[512][4] ;
	float min2 = min[bx] ;
	float max2 = max[bx] ;
 	int i,first_index,last_index,r,position,flag=0,my_K,b;
 	float data ;
 	if(tx<K)
 	{
 	for(i=0;i<4;i++)
	{
		bins1[tx][i]=0 ;
	}
	}
	if(tx<576)
	{
		first_index = tx * 977 ;
		last_index = first_index + 977 ;
	}
	else
	{
		first_index = 576 * 977 + (tx-576) * 976 ;
		last_index = first_index + 976 ;
	}
	position = first_index/250000 ;
	for(i=first_index ; i<last_index;i++)
	{
	data = result[bx * 1000000 + i];
	if(data==max2) 
	r=K-1 ;
	else
	r = floor(((data - min2)/(max2 - min2)) * K);
	atomicAdd(&bins1[r][position],1);
	}
	__syncthreads() ;
	my_K = (tx / 2) ;
	b = my_K < K ;
	if(b)
	bins2[my_K][tx % 2]=bins1[my_K][2*(tx % 2)]+bins1[my_K][2*(tx % 2) + 1] ;
	__syncthreads() ;
	if(((tx % 2)==0) && b)
	bins1[my_K][tx % 2]=bins2[my_K][2*(tx % 2)]+bins2[my_K][2*(tx % 2) + 1] ;
	__syncthreads() ;
	if(((tx % 2)==0) && b)
	{
		hist_gpu[bx * K + my_K] = bins1[my_K][0] ;
	}
}



__global__ void Histogram5(float* result,float* min,float* max,unsigned int K,int* hist_gpu)
{
	__shared__ int bins[1024][2] ;
	float min2 = min[bx] ;
	float max2 = max[bx] ;
 	int i,first_index,last_index,r,position ;
 	float data ;
 	if(tx<K)
 	{
 	for(i=0;i<2;i++)
	{
		bins[tx][i]=0 ;
	}
	}
	if(tx<576)
	{
		first_index = tx * 977 ;
		last_index = first_index + 977 ;
	}
	else
	{
		first_index = 576 * 977 + (tx-576) * 976 ;
		last_index = first_index + 976 ;
	}
	position = first_index/500000 ;
	for(i=first_index ; i<last_index;i++)
	{
	data = result[bx * 1000000 + i];
	if(data==max2) 
	r=K-1 ;
	else
	r = floor(((data - min2)/(max2 - min2)) * K);
	atomicAdd(&bins[r][position],1);
	}
	__syncthreads() ;
	if(tx < K)
	{
		hist_gpu[bx * K + tx] = bins[tx][0]+bins[tx][1] ;
	}

	
}



__global__ void Histogram6(float* result,float* min,float* max,unsigned int K,int* hist_gpu)
{
	__shared__ int bins[5000] ;
	float min2 = min[bx] ;
	float max2 = max[bx] ;
 	int i,first_index,last_index,r ;
 	float data ;
 	if(tx<K)
 	{
		bins[tx]=0 ;
	}
	if(tx<576)
	{
		first_index = tx * 977 ;
		last_index = first_index + 977 ;
	}
	else
	{
		first_index = 576 * 977 + (tx-576) * 976 ;
		last_index = first_index + 976 ;
	}
	for(i=first_index ; i<last_index;i++)
	{
	data = result[bx * 1000000 + i];
	if(data==max2) 
	r=K-1 ;
	else
	r = floor(((data - min2)/(max2 - min2)) * K);
	atomicAdd(&bins[r],1);
	}
	__syncthreads() ;
	for(i=tx;i<K;i+=1024)
	{
		hist_gpu[bx * K + tx] = bins[tx] ;
	}
}





__global__ void put_zero(int* hist_gpu)
{
	hist_gpu[bx*m+tx]=0 ;
}


__global__ void put_zero2(int* result2,unsigned int K)
{
	result2[bx*K*1024+by*1024+1000+tx]=0 ;
}


