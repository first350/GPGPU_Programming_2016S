#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <math.h>
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }
__device__ void update(const char* text, int *pos ,int index, int n)
{
   int v = index + 1;
   char t = text[index] == '\n' ? true : false;
   while (v <= n) {
      //printf("(%d,%d, value: %d) ",index,v,pos[v]);
      if (t){
         //printf("index: %d\n",index);
         pos[v-1] = 1;
      } else {
         return;
      }
      v += v & -v;
   }
   printf("end update\n");
}
__device__ void query(int* pos, int* bit, int index, int n)
{
   int acc = 0;
   int v = index+1;
   v -= v & -v;
   int layer = int(log2(v));
   while(v > 0) {
    if(v%2==1) break;
    if (bit[v-1] == 1-0){
      v = v + int(pow(2,layer));
    }else if(bit[v-1] == 1-1){
      acc += int(pow(2,layer+1));
      v = v - int(pow(2,layer));
    }
    layer -= 1;
   }
   pos[index] = acc;
}
void printInCpu(const char *text, int *pos, int size)
{
   char *debugText = (char*)malloc(size*sizeof(char));
   int *debugPos = (int*)malloc(size*sizeof(int));
   cudaMemcpy(debugText, text, size*sizeof(char), cudaMemcpyDeviceToHost);
   cudaMemcpy(debugPos, pos, size*sizeof(int), cudaMemcpyDeviceToHost);
   for (int i=0; i<size; i++) {
      if (*(debugText+i) == '\n') {
         printf(" ");
      } else {
         printf("%c",*(debugText+i));
      }
   }
   printf("\n");
   for(int i=0;i < size;i++) {
      printf("%d",*(debugPos+i));
   }
   printf("\n");
}
__global__ void BIT(const char* text, int* bit, int n)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if(i < n){
      update(text, bit, i, n);  
   }
}
__global__ void fillPos(int* pos, int* bit, int n)
{
   
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if(i < n){
      query(pos, bit, i, n);  
   }
}
__global__ void count(int* pos, int* bit, char* text, int layer, int n)
{

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int a = text[i] == '\n' ? 0 : 1;
   int b = text[i+layer] == '\n' ? 0 : 1;
   bit[i] = a;
   bit[i+layer] = b;
   if(i < n ){
         if( a*b == 1){
            bit[ i+layer/2 ] = 1;
         } else {
            bit[i+ layer/2] = 0;
         }
   }  

}
void CountPosition(const char *text, int *pos, int text_size)
{
   int* bit = (int*)malloc(sizeof(int)*text_size);
   memset(bit,0,sizeof(int)*text_size);
   int* d_bit;
   cudaMalloc(&d_bit, text_size*sizeof(int));
   cudaMemcpy(d_bit, bit, sizeof(int)*text_size, cudaMemcpyHostToDevice);
   printInCpu(text,d_bit,7);
   printf("%d\n",text_size);
   int layer = 2;
   int numBlocks = int((text_size+255)/256);
   BIT<<<numBlocks,256>>>(text, d_bit, text_size);  
   cudaDeviceSynchronize();
   printInCpu(text, d_bit, text_size);
   fillPos<<<numBlocks,256>>>(pos, d_bit, text_size);
   //printInCpu(text,pos,text_size);
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO

	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
