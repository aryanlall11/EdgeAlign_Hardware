/*--------------------------------------------------------------
CNN Hardware Accelerator IP for layer input dimension of 100 x 4
----------------------------------------------------------------
- Performs 2D convolution between given feature map and kernel
- 3D convolution is performed with the help of MicroBlaze
- Uses PE array based on the Row-stationary approach
- This IP also accounts for padding in the convolution operation

Author: Aryan Lall (17D070053), EE, IIT Bombay
Guide : Prof. Siddharth Tallur, EE, IIT Bombay
--------------------------------------------------------------*/

#include <hls_stream.h>

typedef float T;

const int k = 3;
const int fea_row = 100;
const int fea_col = 4;

const int out_row = fea_row;
const int out_col = fea_col;

static hls::stream<T> stream_fea_in[fea_col];
static hls::stream<T> stream_ker_in[k];
static hls::stream<T> stream_ker[k*out_col - 2];
static hls::stream<T> stream_fea[k*out_col - 2];
static hls::stream<T> stream_acc[(k+1)*out_col - 2];

void ProcessingElement(int pos,
						hls::stream<T> &f_in,
						hls::stream<T> &f_out,
						hls::stream<T> &w_in,
						hls::stream<T> &w_out,
						hls::stream<T> &acc_in,
						hls::stream<T> &acc_out)
{
	T f[fea_row+2], acc_t[out_row], wv, fv;

	f[0] = static_cast<T>(0);
	f[fea_row + 1] = static_cast<T>(0);

	for(int i=1; i<=fea_row; i++)
	{
		#pragma HLS PIPELINE II=1
		f_in.read(fv);
		f_out.write(fv);
		f[i] = fv;
	}

	for(int j=0; j<k; j++)
	{
		w_in.read(wv);
		w_out.write(wv);
		for(int i=0; i<out_row; i++)
		{
			#pragma HLS PIPELINE II=1
			//T prev = (j==0)? ((x<k-1)? acc_in.read(): static_cast<T>(0)): acc_t[i];
			T prev = (j==0)? static_cast<T>(0): acc_t[i];
			acc_t[i] = prev + wv * f[i+j];
			#pragma HLS DEPENDENCE false variable=acc_t
		}
	}

	for(int i=0; i<out_row; i++)
	{
		#pragma HLS PIPELINE II=1
		T v = (pos < ((k-1)*out_col-1))? acc_in.read(): static_cast<T>(0);
		acc_out.write(acc_t[i] + v);
	}
}

void ProcessingElement_n(int pos,
						hls::stream<T> &f_in,
						hls::stream<T> &w_in,
						hls::stream<T> &acc_in,
						hls::stream<T> &acc_out)
{
	T f[fea_row+2], acc_t[out_row], wv, fv;

	f[0] = static_cast<T>(0);
	f[fea_row + 1] = static_cast<T>(0);

	for(int i=1; i<=fea_row; i++)
	{
		#pragma HLS PIPELINE II=1
		f_in.read(fv);
		f[i] = fv;
	}

	for(int j=0; j<k; j++)
	{
		w_in.read(wv);
		for(int i=0; i<out_row; i++)
		{
			#pragma HLS PIPELINE II=1
			//T prev = (j==0)? ((x<k-1)? acc_in.read(): static_cast<T>(0)): acc_t[i];
			T prev = (j==0)? static_cast<T>(0): acc_t[i];
			acc_t[i] = prev + wv * f[i+j];
			#pragma HLS DEPENDENCE false variable=acc_t
		}
	}

	for(int i=0; i<out_row; i++)
	{
		#pragma HLS PIPELINE II=1
		T v = (pos < ((k-1)*out_col-1))? acc_in.read(): static_cast<T>(0);
		acc_out.write(acc_t[i] + v);
	}
}

void ProcessingElement_none(int pos,
						hls::stream<T> &f_in,
						hls::stream<T> &w_in,
						hls::stream<T> &acc_out)
{
	T f[fea_row+2], acc_t[out_row], wv, fv;

	f[0] = static_cast<T>(0);
	f[fea_row + 1] = static_cast<T>(0);

	for(int i=1; i<=fea_row; i++)
	{
		#pragma HLS PIPELINE II=1
		f_in.read(fv);
		f[i] = fv;
	}

	for(int j=0; j<k; j++)
	{
		w_in.read(wv);
		for(int i=0; i<out_row; i++)
		{
			#pragma HLS PIPELINE II=1
			//T prev = (j==0)? ((x<k-1)? acc_in.read(): static_cast<T>(0)): acc_t[i];
			T prev = (j==0)? static_cast<T>(0): acc_t[i];
			acc_t[i] = prev + wv * f[i+j];
			#pragma HLS DEPENDENCE false variable=acc_t
		}
	}

	for(int i=0; i<out_row; i++)
	{
		#pragma HLS PIPELINE II=1
		acc_out.write(acc_t[i]);
	}
}

void ProcessingElement_noacc(int pos,
						hls::stream<T> &f_in,
						hls::stream<T> &f_out,
						hls::stream<T> &w_in,
						hls::stream<T> &w_out,
						hls::stream<T> &acc_out)
{
	T f[fea_row+2], acc_t[out_row], wv, fv;

	f[0] = static_cast<T>(0);
	f[fea_row + 1] = static_cast<T>(0);

	for(int i=1; i<=fea_row; i++)
	{
		#pragma HLS PIPELINE II=1
		f_in.read(fv);
		f_out.write(fv);
		f[i] = fv;
	}

	for(int j=0; j<k; j++)
	{
		w_in.read(wv);
		w_out.write(wv);
		for(int i=0; i<out_row; i++)
		{
			#pragma HLS PIPELINE II=1
			//T prev = (j==0)? ((x<k-1)? acc_in.read(): static_cast<T>(0)): acc_t[i];
			T prev = (j==0)? static_cast<T>(0): acc_t[i];
			acc_t[i] = prev + wv * f[i+j];
			#pragma HLS DEPENDENCE false variable=acc_t
		}
	}

	for(int i=0; i<out_row; i++)
	{
		#pragma HLS PIPELINE II=1
		acc_out.write(acc_t[i]);
	}
}

void ProcessingElement_f_noacc(int pos,
						hls::stream<T> &f_in,
						hls::stream<T> &f_out,
						hls::stream<T> &w_in,
						hls::stream<T> &acc_out)
{
	T f[fea_row+2], acc_t[out_row], wv, fv;

	f[0] = static_cast<T>(0);
	f[fea_row + 1] = static_cast<T>(0);

	for(int i=1; i<=fea_row; i++)
	{
		#pragma HLS PIPELINE II=1
		f_in.read(fv);
		f_out.write(fv);
		f[i] = fv;
	}

	for(int j=0; j<k; j++)
	{
		w_in.read(wv);
		for(int i=0; i<out_row; i++)
		{
			#pragma HLS PIPELINE II=1
			//T prev = (j==0)? ((x<k-1)? acc_in.read(): static_cast<T>(0)): acc_t[i];
			T prev = (j==0)? static_cast<T>(0): acc_t[i];
			acc_t[i] = prev + wv * f[i+j];
			#pragma HLS DEPENDENCE false variable=acc_t
		}
	}

	for(int i=0; i<out_row; i++)
	{
		#pragma HLS PIPELINE II=1
		acc_out.write(acc_t[i]);
	}
}

void ProcessingElement_w(int pos,
						hls::stream<T> &f_in,
						hls::stream<T> &w_in,
						hls::stream<T> &w_out,
						hls::stream<T> &acc_in,
						hls::stream<T> &acc_out)
{
	T f[fea_row+2], acc_t[out_row], wv, fv;

	f[0] = static_cast<T>(0);
	f[fea_row + 1] = static_cast<T>(0);

	for(int i=1; i<=fea_row; i++)
	{
		#pragma HLS PIPELINE II=1
		f_in.read(fv);
		f[i] = fv;
	}

	for(int j=0; j<k; j++)
	{
		w_in.read(wv);
		w_out.write(wv);
		for(int i=0; i<out_row; i++)
		{
			#pragma HLS PIPELINE II=1
			//T prev = (j==0)? ((x<k-1)? acc_in.read(): static_cast<T>(0)): acc_t[i];
			T prev = (j==0)? static_cast<T>(0): acc_t[i];
			acc_t[i] = prev + wv * f[i+j];
			#pragma HLS DEPENDENCE false variable=acc_t
		}
	}

	for(int i=0; i<out_row; i++)
	{
		#pragma HLS PIPELINE II=1
		T v = (pos < ((k-1)*out_col-1))? acc_in.read(): static_cast<T>(0);
		acc_out.write(acc_t[i] + v);
	}
}

static void read_memory(T *ker, T *fea, int D, int depth, int out_c)
{
	Lread_im: for(int i=0;i<fea_col;i++)
		for(int j=0;j<fea_row;j++)
		{
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
			int idx = (i + j*fea_col)*D + depth;
			T v = fea[idx];
			stream_fea_in[i].write(v);
		}

	Lread_ker: for(int i=0;i<k;i++)
		for(int j=0;j<k;j++)
		{
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
			int idx = ((k*out_c + j)*k + i)*D + depth;
			T v = ker[idx];
			stream_ker_in[i].write(v);
		}
}

static void write_result(T *res, T* bias, int out_c, bool nfirst)
{
	T v;
	// Each output channel has a bias value
	T bv = nfirst? static_cast<T>(0): bias[out_c];
	Lstore: for(int i=0; i<out_col; i++)
		for(int j=0; j<out_row; j++)
		{
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_FLATTEN
			int idx = i + j*out_col;
			T res_v = nfirst? static_cast<T>(0): bv;
			stream_acc[i].read(v);
			res[idx] = v + res_v;
		}
}

/* 
The scalar arguments such as depth, out_channels, curr_depth and curr_channel
are used to identify the operand indexes for the 2D convolution. For ex, the 
input feature map is a 3D enitity. The corresponding 2D slice at some iteration
is identified using these indexes. On the frontend side, the MicroBlaze would
iteratively call this IP with varying scalar arguments and the 2D convolution
results would be accumulated to obtain the desired 3D output.
*/
void conv_100x4(T* kernel, T* im, T* bias, T* result, int depth, int out_channels, int curr_depth, int curr_channel)
{
	#pragma HLS INTERFACE m_axi port=kernel bundle=maxi1 offset=slave depth = 512   // Kernel
	#pragma HLS INTERFACE m_axi port=im bundle=maxi1 offset=slave depth = 512       // Image
	// Refer to how bias is used in the convolution operation (TFLite)
	#pragma HLS INTERFACE m_axi port=bias bundle=maxi2 offset=slave depth = 512     // Bias
	#pragma HLS INTERFACE m_axi port=result bundle=maxi2 offset=slave depth = 512   // Result

	#pragma HLS INTERFACE s_axilite port=depth bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=out_channels bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=curr_depth bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=curr_channel bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

	// Maximum depth of FIFOs (*required for co-simulation)
	#pragma HLS STREAM variable = stream_fea depth = 128
	#pragma HLS STREAM variable = stream_acc depth = 128
	#pragma HLS STREAM variable = stream_ker depth = 4
	#pragma HLS STREAM variable = stream_fea_in depth = 128
	#pragma HLS STREAM variable = stream_ker_in depth = 4

	int D = depth;
	int C = out_channels;
	int d = curr_depth;
	int out_c = curr_channel;

	bool nfirst = d>0;

	// Task-level parallelism (refer Xilinx documentation for further details)
	#pragma HLS DATAFLOW

	read_memory(kernel, im, D, d, out_c);

	ProcessingElement_noacc(8, stream_fea_in[1], stream_fea[7], stream_ker_in[2], stream_ker[7], stream_acc[7]);
	ProcessingElement(1, stream_fea_in[0], stream_fea[0], stream_ker_in[1], stream_ker[0], stream_acc[7], stream_acc[0]);
	ProcessingElement_noacc(9, stream_fea_in[2], stream_fea[8], stream_ker[7], stream_ker[8], stream_acc[8]);
	ProcessingElement_f_noacc(10, stream_fea_in[3], stream_fea[9], stream_ker[8], stream_acc[9]);
	ProcessingElement(5, stream_fea[7], stream_fea[4], stream_ker[0], stream_ker[4], stream_acc[8], stream_acc[4]);
	ProcessingElement_w(2, stream_fea[0], stream_ker_in[0], stream_ker[1], stream_acc[4], stream_acc[1]);
	ProcessingElement(6, stream_fea[8], stream_fea[5], stream_ker[4], stream_ker[5], stream_acc[9], stream_acc[5]);
	ProcessingElement_none(7, stream_fea[9], stream_ker[5], stream_acc[6]);
	ProcessingElement_w(3, stream_fea[4], stream_ker[1], stream_ker[2], stream_acc[5], stream_acc[2]);
	ProcessingElement_n(4, stream_fea[5], stream_ker[2], stream_acc[6], stream_acc[3]);

	write_result(result, bias, out_c, nfirst);
}
