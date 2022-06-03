#include <hls_stream.h>

typedef float T;            // Floating precision

#define N 15                // 15 x 15 image
#define k 3                 // 3 x 3 kernel
#define out_size N-k+1      // Stride = 1, No padding

hls::stream<T> stream_in[N];                // Input stream at boundaries for feature rows
hls::stream<T> stream_ker[k*(out_size+1)];  // Intermediate kernel streams
hls::stream<T> stream_fea[k*out_size];      // Intermediate feature streams
hls::stream<T> stream_acc[(k+1)*out_size];  // Intermediate accumulator streams

void ProcessingElement(int x, int y,            // Define the functionality of PE
						hls::stream<T> &w_in,
						hls::stream<T> &w_out,
						hls::stream<T> &f_in,
						hls::stream<T> &f_out,
						hls::stream<T> &acc_in,
						hls::stream<T> &acc_out)
{
	T w[k], f[N], acc_t[out_size], wv, fv;

	for(int i=0; i<N; i++)
	{
		#pragma HLS PIPELINE II=1
		f_in >> fv;           // Read feature element from stream
		f_out << fv;          // Write feature element to the stream of neighbouring PE
		f[i] = fv;
	}

	for(int j=0; j<k; j++)
	{
		w_in >> wv;			  // Read kernel element from stream
		w_out << wv;		  // Write kernel element to the stream of neighbouring PE
		for(int i=0; i<out_size; i++)
		{
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_FLATTEN
			T prev = (j==0)? static_cast<T>(0): acc_t[i];
			// MAC operation
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
	// Read kernel elements from the memory using m_axi interface
	Lread_ker: for(int i=0;i<k;i++)
		for(int j=0;j<k;j++)
		{
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
			const auto v = ker[i*k + j];
			// Write onto boundary kernel stream interfaces
			stream_ker[i*(out_size+1)] << v;
		}

	// Read feature elements from the memory using m_axi interface
	Lread_im: for(int i=0;i<N;i++)
		for(int j=0;j<N;j++)
		{
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
			const auto v = fea[i*N + j];
			// Write onto boundary input stream interfaces
			stream_in[i] << v;
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
			stream_acc[i] >> v;     // Read from top-most accumulator stream interface
			res[i*out_size + j] = v;
		}
}

// Top-level module
void conv_pe(T* kernel, T* im, T* result)
{
	#pragma HLS INTERFACE m_axi port=kernel bundle=maxi1 offset=slave  // Kernel
	#pragma HLS INTERFACE m_axi port=im bundle=maxi2 offset=slave      // Image
	#pragma HLS INTERFACE m_axi port=result bundle=maxi3 offset=slave  // Result
	#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

	// Maximum depth of FIFOs (*required for co-simulation)
	// Ex - stream_fea can hold maximum 16 elements > N
	#pragma HLS STREAM variable = stream_fea depth = 16
	#pragma HLS STREAM variable = stream_acc depth = 16
	#pragma HLS STREAM variable = stream_ker depth = 8
	#pragma HLS STREAM variable = stream_in depth = 16

	// Task-level parallelism (refer Xilinx documentation for further details)
	#pragma HLS DATAFLOW

	//----------------------------------------------------------------
	read_memory(kernel, im);    // Read operands from memory and write onto appropriate streams

	// The following function calls are hardcoded since we can't obtain a general formulae
	// These function calls can be generated using a Python file (PE.py) attached in the same folder
	
	ProcessingElement(0, 0, stream_ker[0], stream_ker[1], stream_in[0], stream_fea[0], stream_acc[13], stream_acc[0]);
	ProcessingElement(0, 1, stream_ker[1], stream_ker[2], stream_fea[13], stream_fea[1], stream_acc[14], stream_acc[1]);
	ProcessingElement(0, 2, stream_ker[2], stream_ker[3], stream_fea[14], stream_fea[2], stream_acc[15], stream_acc[2]);
	ProcessingElement(0, 3, stream_ker[3], stream_ker[4], stream_fea[15], stream_fea[3], stream_acc[16], stream_acc[3]);
	ProcessingElement(0, 4, stream_ker[4], stream_ker[5], stream_fea[16], stream_fea[4], stream_acc[17], stream_acc[4]);
	ProcessingElement(0, 5, stream_ker[5], stream_ker[6], stream_fea[17], stream_fea[5], stream_acc[18], stream_acc[5]);
	ProcessingElement(0, 6, stream_ker[6], stream_ker[7], stream_fea[18], stream_fea[6], stream_acc[19], stream_acc[6]);
	ProcessingElement(0, 7, stream_ker[7], stream_ker[8], stream_fea[19], stream_fea[7], stream_acc[20], stream_acc[7]);
	ProcessingElement(0, 8, stream_ker[8], stream_ker[9], stream_fea[20], stream_fea[8], stream_acc[21], stream_acc[8]);
	ProcessingElement(0, 9, stream_ker[9], stream_ker[10], stream_fea[21], stream_fea[9], stream_acc[22], stream_acc[9]);
	ProcessingElement(0, 10, stream_ker[10], stream_ker[11], stream_fea[22], stream_fea[10], stream_acc[23], stream_acc[10]);
	ProcessingElement(0, 11, stream_ker[11], stream_ker[12], stream_fea[23], stream_fea[11], stream_acc[24], stream_acc[11]);
	ProcessingElement(0, 12, stream_ker[12], stream_ker[13], stream_fea[24], stream_fea[12], stream_acc[25], stream_acc[12]);
	ProcessingElement(1, 0, stream_ker[14], stream_ker[15], stream_in[1], stream_fea[13], stream_acc[26], stream_acc[13]);
	ProcessingElement(1, 1, stream_ker[15], stream_ker[16], stream_fea[26], stream_fea[14], stream_acc[27], stream_acc[14]);
	ProcessingElement(1, 2, stream_ker[16], stream_ker[17], stream_fea[27], stream_fea[15], stream_acc[28], stream_acc[15]);
	ProcessingElement(1, 3, stream_ker[17], stream_ker[18], stream_fea[28], stream_fea[16], stream_acc[29], stream_acc[16]);
	ProcessingElement(1, 4, stream_ker[18], stream_ker[19], stream_fea[29], stream_fea[17], stream_acc[30], stream_acc[17]);
	ProcessingElement(1, 5, stream_ker[19], stream_ker[20], stream_fea[30], stream_fea[18], stream_acc[31], stream_acc[18]);
	ProcessingElement(1, 6, stream_ker[20], stream_ker[21], stream_fea[31], stream_fea[19], stream_acc[32], stream_acc[19]);
	ProcessingElement(1, 7, stream_ker[21], stream_ker[22], stream_fea[32], stream_fea[20], stream_acc[33], stream_acc[20]);
	ProcessingElement(1, 8, stream_ker[22], stream_ker[23], stream_fea[33], stream_fea[21], stream_acc[34], stream_acc[21]);
	ProcessingElement(1, 9, stream_ker[23], stream_ker[24], stream_fea[34], stream_fea[22], stream_acc[35], stream_acc[22]);
	ProcessingElement(1, 10, stream_ker[24], stream_ker[25], stream_fea[35], stream_fea[23], stream_acc[36], stream_acc[23]);
	ProcessingElement(1, 11, stream_ker[25], stream_ker[26], stream_fea[36], stream_fea[24], stream_acc[37], stream_acc[24]);
	ProcessingElement(1, 12, stream_ker[26], stream_ker[27], stream_fea[37], stream_fea[25], stream_acc[38], stream_acc[25]);
	ProcessingElement(2, 0, stream_ker[28], stream_ker[29], stream_in[2], stream_fea[26], stream_acc[39], stream_acc[26]);
	ProcessingElement(2, 1, stream_ker[29], stream_ker[30], stream_in[3], stream_fea[27], stream_acc[40], stream_acc[27]);
	ProcessingElement(2, 2, stream_ker[30], stream_ker[31], stream_in[4], stream_fea[28], stream_acc[41], stream_acc[28]);
	ProcessingElement(2, 3, stream_ker[31], stream_ker[32], stream_in[5], stream_fea[29], stream_acc[42], stream_acc[29]);
	ProcessingElement(2, 4, stream_ker[32], stream_ker[33], stream_in[6], stream_fea[30], stream_acc[43], stream_acc[30]);
	ProcessingElement(2, 5, stream_ker[33], stream_ker[34], stream_in[7], stream_fea[31], stream_acc[44], stream_acc[31]);
	ProcessingElement(2, 6, stream_ker[34], stream_ker[35], stream_in[8], stream_fea[32], stream_acc[45], stream_acc[32]);
	ProcessingElement(2, 7, stream_ker[35], stream_ker[36], stream_in[9], stream_fea[33], stream_acc[46], stream_acc[33]);
	ProcessingElement(2, 8, stream_ker[36], stream_ker[37], stream_in[10], stream_fea[34], stream_acc[47], stream_acc[34]);
	ProcessingElement(2, 9, stream_ker[37], stream_ker[38], stream_in[11], stream_fea[35], stream_acc[48], stream_acc[35]);
	ProcessingElement(2, 10, stream_ker[38], stream_ker[39], stream_in[12], stream_fea[36], stream_acc[49], stream_acc[36]);
	ProcessingElement(2, 11, stream_ker[39], stream_ker[40], stream_in[13], stream_fea[37], stream_acc[50], stream_acc[37]);
	ProcessingElement(2, 12, stream_ker[40], stream_ker[41], stream_in[14], stream_fea[38], stream_acc[51], stream_acc[38]);

	write_result(result);      // Store the final convolution result back into the memery
	//----------------------------------------------------------------
}

