#include <hls_stream.h>

typedef float T;            // Floating precision

const int N = 15;                // 15 x 15 image
const int k = 3;                 // 3 x 3 kernel
const int out_size = N-k+1;      // Stride = 1, No padding

static hls::stream<T> stream_fea_in[N];             // Boundary streams for feature row inputs
static hls::stream<T> stream_ker_in[k];			    // Boundary streams for kernel row inputs
static hls::stream<T> stream_ker[k*out_size];       // Input kernel stream for each PE
static hls::stream<T> stream_fea[k*out_size];       // Input feature stream for each PE
static hls::stream<T> stream_acc[(k+1)*out_size];   // Intermediate accumulator streams

void ProcessingElement(int x, int y,              // Define the functionality of PE
						hls::stream<T> &w_in,
						hls::stream<T> &f_in,
						hls::stream<T> &acc_in,
						hls::stream<T> &acc_out)
{
	T f[N], acc_t[out_size], wv;

	for(int i=0; i<N; i++)
	{
		#pragma HLS PIPELINE II=1
		f_in.read(f[i]);               // Read feature element from stream
	}

	for(int j=0; j<k; j++)
	{
		w_in.read(wv);                 // Read kernel element from stream
		for(int i=0; i<out_size; i++)
		{
			#pragma HLS PIPELINE II=1
			T prev = (j==0)? static_cast<T>(0): acc_t[i];
			acc_t[i] = prev + wv * f[i+j];
			#pragma HLS DEPENDENCE false variable=acc_t
		}
	}

	for(int i=0; i<out_size; i++)
	{
		#pragma HLS PIPELINE II=1
		// Accumulation of partial convolution results in the upward direction
		T v = (x<k-1)? acc_in.read(): static_cast<T>(0);
		acc_out.write(acc_t[i] + v);
	}
}

static void read_memory(T *ker, T *fea)
{
	// Read feature elements from the memory using m_axi interface
	Lread_im: for(int i=0;i<N;i++)
		for(int j=0;j<N;j++)
		{
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
			// Write onto boundary feature stream interfaces
			T v = fea[i*N + j];
			stream_fea_in[i].write(v);
		}

	// Read kernel elements from the memory using m_axi interface
	Lread_ker: for(int i=0;i<k;i++)
		for(int j=0;j<k;j++)
		{
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
			// Write onto boundary kernel stream interfaces
			T v = ker[i*k + j];
			stream_ker_in[i].write(v);
		}
}

static void ker_router(int x, hls::stream<T> &w)     // Define the functionality of Kernel Router
{
	T v;
	for(int i=0; i<k; i++)
	{
		w.read(v);
		for(int j=0; j<out_size; j++)
		{
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
			stream_ker[x*out_size + j].write(v);   // Stream along all PEs in the horizontal direction
		}
	}
}

void write_stream(T v[], hls::stream<T> &f)
{
	for(int j=0; j<N; j++)
	{
		#pragma HLS PIPELINE II=1
		f.write(v[j]);
	}
}

static void fea_router(int row, hls::stream<T> &f)   // Define the functionality of Feature Router
{
	T v, val[N];
	bool c = row<k;
	/* The selection of the PEs can be guided through a well-structured formulae 
	   based on the stride value, etc. The following code is written for stride = 1.
	*/
	int iter = c? (row+1): ((row>=out_size)? (N-row): k);
	int x_f = c? row: k-1;
	int y_f = c? 0: (row-k+1);
	int idx = x_f*out_size + y_f;

	for(int i=0; i<N; i++)
	{
		#pragma HLS PIPELINE II=1
		f.read(v);
		val[i] = v;
	}
	for(int i=0; i<iter; i++)
	{
		#pragma HLS UNROLL
		// Write feature elements on the input feature stream of the selected PE
		write_stream(val, stream_fea[idx]);
		idx = idx - out_size + 1;
	}
}

static void write_result(T *res)
{
	T v;
	// Write output elements into the memory using m_axi interface
	Lstore: for(int i=0; i<out_size; i++)
		for(int j=0; j<out_size; j++)
		{
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_FLATTEN
			stream_acc[i].read(v);     // Read from top-most accumulator stream interface
			res[i*out_size + j] = v;
		}
}

// Top-level module
void conv_pe(T* kernel, T* im, T* result)
{
	#pragma HLS INTERFACE m_axi port=kernel bundle=maxi1 offset=slave depth = 512   // Kernel
	#pragma HLS INTERFACE m_axi port=im bundle=maxi2 offset=slave depth = 512       // Image
	#pragma HLS INTERFACE m_axi port=result bundle=maxi3 offset=slave depth = 512   // Result
	#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

	// Maximum depth of FIFOs (*required for co-simulation)
	// Ex - stream_fea can hold maximum 16 elements > N
	#pragma HLS STREAM variable = stream_fea depth = 16
	#pragma HLS STREAM variable = stream_acc depth = 16
	#pragma HLS STREAM variable = stream_ker depth = 8
	#pragma HLS STREAM variable = stream_fea_in depth = 16
	#pragma HLS STREAM variable = stream_ker_in depth = 8

	// Task-level parallelism (refer Xilinx documentation for further details)
	#pragma HLS DATAFLOW

	//----------------------------------------------------------------
	read_memory(kernel, im);    // Read operands from memory and write onto appropriate streams

	for(int i=N-1; i>=0; i--)
	{
		#pragma HLS UNROLL   // All feature routers working in parallel
		fea_router(i, stream_fea_in[i]);
	}

	for(int i=k-1; i>=0; i--)
	{
		#pragma HLS UNROLL   // All kernel routers working in parallel
		ker_router(i, stream_ker_in[i]);
	}

	for(int i=k*out_size-1; i>=0; i--)
	{
		#pragma HLS UNROLL   // All PEs working in parallel
		ProcessingElement(i/out_size, 0, stream_ker[i], stream_fea[i], stream_acc[i + out_size], stream_acc[i]);
	}

	write_result(result);
	//----------------------------------------------------------------
}
