#include <cstdio>
#include <cstdlib>
#include "../utils/SyncedMemory.h"
#include <string>
#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

__global__ void SomeTransform(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < fsize and input_gpu[idx] != '\n') {
		if( input_gpu[idx] >= 97) {
			input_gpu[idx] -= 32;
		} else {
			input_gpu[idx] += 32;
		}
	
	}
}

int main(int argc, char **argv)
{
	// init, and check
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if (not fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	// get file size
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// read files
	MemoryBuffer<char> text(fsize+1);
	auto text_smem = text.CreateSync(fsize);
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);

	char *input_gpu = text_smem.get_gpu_rw();
	// chage first 5*32 character , swap lowercase and uppercase, change space to @
	// Don't transform over the tail
	// And don't transform the line breaks
	SomeTransform<<<5, 32>>>(input_gpu, fsize);

	puts(text_smem.get_cpu_ro());
	return 0;
}
