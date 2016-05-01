#include "lab3.h"
#include <cstdio>
#include <math.h>
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}
__global__ void CalculateBorder(
        float* e,
        float* totalError,
	const float *mask,
	int *border,
	float* R,
	float*X,
	float*nextX,
	float*D,
	float*b,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	border[curt] = 0;
	e[curt] = 0.0;
	e[curt*3+1] = 0.0;
	e[curt*3+2] = 0.0;
	totalError[curt] = 0.0;
	    R[curt*4+0] = -1.0;
	    R[curt*4+1] = -1.0;
	    R[curt*4+2] = -1.0;
	    R[curt*4+3] = -1.0;
	    X[curt*3+0] = 0.0;
	    X[curt*3+1] = 0.0;
	    X[curt*3+2] = 0.0;
	    b[curt*3+0] = 0.0;
	    b[curt*3+1] = 0.0;
	    b[curt*3+2] = 0.0;
	    nextX[curt*3+0] = 0.0;
	    nextX[curt*3+1] = 0.0;
	    nextX[curt*3+2] = 0.0;
	    D[curt] = 0.0;
	int dp[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
	    R[curt*4+0] = -1.0;
	    R[curt*4+1] = -1.0;
	    R[curt*4+2] = -1.0;
	    R[curt*4+3] = -1.0;
	    X[curt*3+0] = 0.0;
	    X[curt*3+1] = 0.0;
	    X[curt*3+2] = 0.0;
	    b[curt*3+0] = 0.0;
	    b[curt*3+1] = 0.0;
	    b[curt*3+2] = 0.0;
	    nextX[curt*3+0] = 0.0;
	    nextX[curt*3+1] = 0.0;
	    nextX[curt*3+2] = 0.0;
	    D[curt] = 0.0;
		for(int k=0;k<4;k++){
			int ytt = yt + dp[k][1];
			int xtt = xt + dp[k][0];
			int curtt = wt*ytt+xtt; // [x2][y2]
			if (ytt < 0 ||  mask[curtt] < 127.0f) {
				border[curt] = -1;
				return;
			}
		}
	}


}
__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	float *R,
	float *D,
	float *X,
	float *b,
	int* border,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	int Np = 0;
	if (yt > ht || xt > wt || mask[curt] < 127.0f)
		return;
	// const int yb = oy+yt, xb = ox+xt;
	// const int curb = wb*yb+xb;
	float pR = target[curt*3+0];
	float pG = target[curt*3+1];
	float pB = target[curt*3+2];
	int dp[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
	b[curt*3+0] = 0.0;
	b[curt*3+1] = 0.0;
	b[curt*3+2] = 0.0;
	// in selection area
	if (mask[curt] > 127.0f) {
		for(int k=0; k<4 ; k++) {
			int ytt = yt + dp[k][1];
			int xtt = xt + dp[k][0];
			int curtt = wt*ytt+xtt; // [x2][y2]
			if (ytt<0||xtt<0||mask[curtt]<127.0f)
				continue;
			Np += 1;
			R[curt*4+k] = -1;// init R
			int index = border[curtt];
			if (index == -1) {
				int yb = oy+ytt, xb = ox+xtt;
				int curb = wb*yb+xb;
				//it's border pixel
				b[curt*3+0] += background[curb*3+0];
				b[curt*3+1] += background[curb*3+1];
				b[curt*3+2] += background[curb*3+2];
			} else {
				R[curt*4+k] = curtt;
				float qR = target[curtt*3+0];
				float qG = target[curtt*3+1];
				float qB = target[curtt*3+2];
				b[curt*3+0] += pR-qR;
				b[curt*3+1] += pG-qG;
				b[curt*3+2] += pB-qB;
			}
		}
		D[curt] = (float)Np;
	}

}
__global__ void getError(
        const float* mask,
        float* totalError,
        float* e,
	float *R,
	float *D,
	float *X,
	float *b,
	float* nextX,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt > ht || xt > wt || mask[curt] < 127.0f) {
	 return; 
        }
        float total = 0.0;
        for(int n=0;n<4;n++) {
            e[curt*3+0] = b[curt*3+0];
            e[curt*3+1] = b[curt*3+1];
            e[curt*3+2] = b[curt*3+2];
            if(R[curt*4+n] >= 0) {
               int index = (int)R[curt*4+n];
               for(int k=0;k<3;k++) {
                  e[curt*3+k] += X[index*3+k];   
               }
            }
        }
        e[curt*3+0] -= D[curt]*X[curt*3+0];
        e[curt*3+1] -= D[curt]*X[curt*3+1];
        e[curt*3+2] -= D[curt]*X[curt*3+2];
        total = (e[curt*3+0]*e[curt*3+0]+e[curt*3+1]*e[curt*3+1]+e[curt*3+2]*e[curt*3+2]);
        totalError[curt] = total;
        
}
__global__ void Jacobi(
        const float* mask,
	float *R,
	float *D,
	float *X,
	float *b,
	float* nextX,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt > ht || xt > wt || mask[curt] < 127.0f)
		return;
	for(int k=0;k<3;k++) {
		nextX[curt*3+k] = b[curt*3+k];
	}
	for(int n=0;n<4;n++) {
		if (R[curt*4+n] >= 0) {
			int index = (int)R[curt*4+n];
			for(int k=0;k<3;k++) {
				nextX[curt*3+k] += X[index*3+k];
			}
		}
	}
	for(int k=0;k<3;k++) {
		nextX[curt*3+k] /= D[curt];
		//printf("nextX: %f\n",nextX[curt*3+k]);
	}
	/*
	if(D[curt] > 0) {
		printf("%f\n",D[curt]);
	}
	*/

}
void printfA(float*R,const int wt,const int ht) {
	float* buf = (float*)malloc(sizeof(float)*wt*ht*3);
	cudaMemcpy(buf, R, 3*wt*ht*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0;i<wt*ht*3;i++){
	    if(buf[i] != 0)
		printf("r: %f \n",buf[i]);
	}
}
void printMask(	int *border ) {
	int* buf = (int*)malloc(sizeof(int)*82944);
	cudaMemcpy(buf, border, 82944*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0;i<82944;i++){
		printf("%d \n",buf[i]);
	}

}
void showError(float*e,int wt,int ht ) {
	float* buf = (float*)malloc(sizeof(float)*wt*ht);
	cudaMemcpy(buf, e, wt*ht*sizeof(float), cudaMemcpyDeviceToHost);
	float total = 0.0;
	for(int i=0;i<wt*ht;i++){
	 total += buf[i];
	 //printf("buf: %f\n",buf[i]);
        }
        printf("total: %f\n",sqrt(total));
}
void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	float *fixed, *A, *B;
	float *R,*D,*X,*b,*nextX,*e,*totalError;
	int *border;
	cudaMalloc(&fixed, 3 * wt * ht * sizeof(float));
	cudaMalloc(&border, wt * ht * sizeof(int));
	cudaMalloc(&A, 3 * wt * ht * sizeof(float));
	cudaMalloc(&e, 3 * wt * ht * sizeof(float));
	cudaMalloc(&totalError, wt * ht * sizeof(float));
	cudaMalloc(&B, 3 * wt * ht * sizeof(float));
	cudaMalloc(&R, 4 * wt * ht * sizeof(float)); // remain
	cudaMalloc(&D, wt * ht * sizeof(float)); // diagonal
	cudaMalloc(&X, 3 * wt * ht * sizeof(float));
	cudaMalloc(&nextX, 3 * wt * ht * sizeof(float));
	cudaMalloc(&b, 3 * wt * ht * sizeof(float));
	//printMask(mask);
	//exit(1);
	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)) ,bdim(32,16);
	CalculateBorder<<< gdim, bdim>>>(
		e,totalError,mask, border,R,X,nextX,D,b,
		wb, hb, wt, ht, oy, ox
	);
	//printMask(border);
	//exit(1);
	CalculateFixed<<< gdim, bdim>>>(
		background, target, mask, fixed,R,D,X,b,border,
		wb, hb, wt, ht, oy, ox
	);
	//printfA(X,wt,ht);
	//exit(1);
	for (int i=0;i<20000;i++) {
		Jacobi<<< gdim, bdim>>>(
			mask,R,D,X,b,nextX,wb, hb, wt, ht, oy, ox
		);
		
		Jacobi<<< gdim, bdim>>>(
			mask,R,D,nextX,b,X,wb, hb, wt, ht, oy, ox
		);

               getError<<< gdim, bdim>>>(
			mask,totalError,e,R,D,X,b,nextX,wb, hb, wt, ht, oy, ox
		);
	       showError(totalError,wt,ht);
	}
	printfA(X,wt,ht);

        cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, X, mask, output,
		wb, hb, wt, ht, oy, ox
	);
}
