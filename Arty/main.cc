#include "xparameters.h"
#include "xuartlite.h"
#include "xuartlite_l.h"
#include "xil_printf.h"
#include "xtmrctr.h"
#include <stdio.h>

#include <string.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "sleep.h"
#include "edgealign_model.h"
#include "peripheral.h"

static XUartLite UART0_instance;

const int MAX_LEN = 1500;
const int window = 50;
const int pixel_size = 2;

const float BP[5][6] = {{0, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 1}, {1, 1, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};
const int action_len = 1000;

typedef struct {
	XTmrCtr m_AxiTimer;
	unsigned int m_tickCounter1;
	unsigned int m_tickCounter2;
	double m_clockPeriodmSeconds;
	double m_timerClockFreq;
}timer;
timer tm;

unsigned int startTimer() {
	// Start timer 0 (There are two, but depends how you configured in vivado)
	XTmrCtr_Reset(&tm.m_AxiTimer,0);
	tm.m_tickCounter1 =  XTmrCtr_GetValue(&tm.m_AxiTimer, 0);
	XTmrCtr_Start(&tm.m_AxiTimer, 0);
	return tm.m_tickCounter1;
}

unsigned int stopTimer() {
	XTmrCtr_Stop(&tm.m_AxiTimer, 0);
	tm.m_tickCounter2 =  XTmrCtr_GetValue(&tm.m_AxiTimer, 0);
	return tm.m_tickCounter2 - tm.m_tickCounter1;
}

float getElapsedTimerInmSeconds() {
	float elapsedTimeInmSeconds = (float)(tm.m_tickCounter2 - tm.m_tickCounter1) * tm.m_clockPeriodmSeconds;
	return elapsedTimeInmSeconds;
}

void readSeq(uint8_t *data, int length)
{
	for(int i=0; i<length; i++)
    {
		data[i] = XUartLite_RecvByte(XPAR_UARTLITE_0_BASEADDR);
    }
}

void sendSeq(uint8_t *data, int length)
{
	print("rc\n");
	xil_printf("%d\n", length);

	for(int i=0; i<length; i++)
	{
		//xil_printf("%d\n", int(data[i]));
		XUartLite_SendByte(XPAR_UARTLITE_0_BASEADDR, data[i]);
	}
}

float receive_float(void)
{
	union{
		float val_float;
		unsigned char bytes[4];
	}data;
	for(int i=0; i<4; i++)
		data.bytes[i] = XUartLite_RecvByte(XPAR_UARTLITE_0_BASEADDR);
	return data.val_float;
}

void fill_states(float *arr, int seed, int val)
{
    for(int i=0; i<6; i++)
    {
        int idx = seed + i;
        float value = float(BP[val][i]);
        arr[idx] = value;
        arr[idx + 12] = value;
    }
}

//XStatus readUartBytes(int8_t *data, float scale, int32_t zero_point)
//{
//	uint8_t rxBuffer;
//	float temp;
//    for(int i=0;i<imagesize;i++)
//    {
//    	rxBuffer = XUartLite_RecvByte(XPAR_UARTLITE_0_BASEADDR);
//    	temp = static_cast<float>(rxBuffer) / 255.0f;
//    	data[i] = temp/scale + zero_point;
//    }
//}

/*void init_fc()
{
	print("Initializing Fully connected module...\n");
	fc_cfg = XFullyconnected_LookupConfig(XPAR_FULLYCONNECTED_0_DEVICE_ID);
	if(fc_cfg)
	{
		int status = XFullyconnected_CfgInitialize(&fc, fc_cfg);
		if(status != XST_SUCCESS)
			print("Error Initializing Multiply module!\n");
	}
}*/

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by compiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 120 * 1024;
  __attribute__((aligned(16)))uint8_t tensor_arena[kTensorArenaSize];
} //

int main()
{
	microblaze_enable_icache();
	microblaze_enable_dcache();
	microblaze_invalidate_dcache();
	/*------------------------------ INITIALIZATION ---------------------------------*/
	int status = XUartLite_Initialize(&UART0_instance, XPAR_AXI_UARTLITE_0_DEVICE_ID);
	InitPeripherals();

	XTmrCtr_Initialize(&tm.m_AxiTimer, XPAR_TMRCTR_0_DEVICE_ID);  //Initialize Timer
	tm.m_timerClockFreq = (double) XPAR_AXI_TIMER_0_CLOCK_FREQ_HZ;
	tm.m_clockPeriodmSeconds = (double)1000/tm.m_timerClockFreq;
	/*-------------------------------------------------------------------------------*/
	print("Hello World\r\n");

	uint8_t seq1[MAX_LEN], seq2[MAX_LEN], actions[action_len];
	float total_time = 0;

	TfLiteStatus tflite_status;

	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	error_reporter->Report("STM32 Disco TFLite EdgeAlign");

	model = tflite::GetModel(edgealign_model_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION)
	{
	 error_reporter->Report("Model version does not match Schema");
	 while(1);
	}

	static tflite::MicroMutableOpResolver<4> micro_op_resolver;

	tflite_status = micro_op_resolver.AddConv2D();
	if (tflite_status != kTfLiteOk)
	{
		error_reporter->Report("Could not add CONVOLUTION op");
		while(1);
	}

	tflite_status = micro_op_resolver.AddFullyConnected();
	if (tflite_status != kTfLiteOk)
	{
		error_reporter->Report("Could not add FULLY CONNECTED op");
		while(1);
	}

	tflite_status = micro_op_resolver.AddMaxPool2D();
	if (tflite_status != kTfLiteOk)
	{
		error_reporter->Report("Could not add MAXPOOL op");
		while(1);
	}

	tflite_status = micro_op_resolver.AddReshape();
	if (tflite_status != kTfLiteOk)
	{
	   error_reporter->Report("Could not add FLATTEN op");
	   while(1);
	}

	static tflite::MicroInterpreter static_interpreter(
		model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
	interpreter = &static_interpreter;

	tflite_status = interpreter->AllocateTensors();
	if (tflite_status != kTfLiteOk)
	{
	  error_reporter->Report("AllocateTensors() failed");
	  while(1);
	}

	// Assign model input and output buffers (tensors) to pointers
	model_input = interpreter->input(0);
	model_output = interpreter->output(0);

	int num_elements = model_input->bytes / sizeof(float);
	xil_printf("Number of input elements: %d\r\n", num_elements);

	xil_printf("start!\n");

	while (1)
	  {
		total_time = 0;
		/* USER CODE END WHILE */
		int l1 = int(receive_float()); // Length of sequence 1
		int l2 = int(receive_float()); // Length of sequence 2

		error_reporter->Report("Reading sub-sequences...");

		readSeq(seq1, l1);  // Read sequence 1
		readSeq(seq2, l2);  // Read sequence 2

		error_reporter->Report("Reading completed!");

		int x = 0;   // Seq1 index
		int y = 0;   // Seq2 index

		int total_actions = 0;
		int sub_actions = 0;

		error_reporter->Report("Starting sub-alignment...");

		while(x < l1 && y < l2)
		{
			// Start timer
			startTimer();

			int seed1 = 0;
			int seed2 = 6;
			// Constuct the State
			//error_reporter->Report("Filling states...");
			for(int i=0; i<window; i++)
			{
				int s1_val = (x + i) < l1? seq1[x+i]: 4;
				fill_states(model_input->data.f, seed1, s1_val);
				seed1 += 24;
				int s2_val = (y + i) < l2? seq2[y+i]: 4;
				fill_states(model_input->data.f, seed2, s2_val);
				seed2 += 24;
			}
			//error_reporter->Report("Start invoke");

			// Model Inference ~ Next action
			tflite_status = interpreter->Invoke();
			if (tflite_status != kTfLiteOk)
			{
			  error_reporter->Report("Invoke failed");
			}
			//error_reporter->Report("Invoke done");
			// Get action from Q-values
			float max = model_output->data.f[1];
			int action = 1;
			for(int i=2; i<4; i++)
			{
				float val = model_output->data.f[i];
				if(max<val)
				{
					max = val;
					action = i;
				}
			}
			// Take action
			if(action == 1)      // Match
			{
				x += 1;
				y += 1;
			}
			else if(action == 2) // Insert
				y += 1;
			else
				x += 1;          // Delete

			// Stop timer
			stopTimer();

			total_time += getElapsedTimerInmSeconds();

			actions[sub_actions] = action;
			total_actions += 1;
			sub_actions += 1;

			if(sub_actions == action_len)
			{
				sendSeq(actions, action_len);
				sub_actions = 0;
			}
		}

		if(sub_actions > 0)
			sendSeq(actions, sub_actions);

		error_reporter->Report("Alignment completed!");

		xil_printf("Total actions: %d | Duration(s): %d\r\n",
				  total_actions,
				  int(total_time/1000));

		xil_printf("done!\n");
	  }

	return 0;
}

extern "C" void DebugLog(const char* s)
{
	print(s);
}
