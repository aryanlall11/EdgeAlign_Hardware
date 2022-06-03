#include <stdio.h>
#include <cstdlib>

typedef float T;

const int N = 15;
const int ker = 3;
const int out_size = N-ker+1;

void conv_pe(T* kernel, T* im, T* result);

int main()
{
	T kernel[ker*ker]= {-1, 0, 1, -1, 0, 1, -1, 0, 1};

	T im[N*N];

	T result[out_size*out_size];

	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{
			im[i*N + j] = static_cast<T>(i+j);
		}
	}

	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{
			printf("%d ", int(im[i*N+j]));
		}
		printf("\n");
	}

	conv_pe((T *)kernel, (T *)im, (T *)result);

	for(int i=0;i<out_size;i++)
	{
		for(int j=0;j<out_size;j++)
		{
			printf("%d ", int(result[i*out_size+j]));
		}
		printf("\n");
	}

	return 0;
}
