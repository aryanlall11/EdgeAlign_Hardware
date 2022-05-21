#include "main.h"
#include <stdio.h>
#include <string.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "mnist_model.h"
#include "edgealign_model.h"

/* USER CODE END Includes */

TIM_HandleTypeDef htim14;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */
// TFLite globals
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
}
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM14_Init(void);
static void MX_USART1_UART_Init(void);
//static void readUartBytes(float *data, int imageSize);
static float receive_float(void);
static void readSeq(uint8_t *data, int length);
static void fill_states(float *arr, int seed, int val);
static void sendSeq(uint8_t *data, int length);
/*----------------------------------------------------------------------------*/

#define MAX_LEN 1500
#define window 50
#define pixel_size 2

const float BP[5][6] = {{0, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 1}, {1, 1, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}};
char buf[50];
int buf_len = 0;

const int action_len = 1000;

int main(void)
{
  /* USER CODE BEGIN 1 */
  TfLiteStatus tflite_status;
  uint32_t num_elements;
  uint16_t timestamp;
  float total_time = 0;

  uint8_t seq1[MAX_LEN], seq2[MAX_LEN], actions[action_len];
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM14_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */
  HAL_TIM_Base_Start(&htim14);

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

    // Build an interpreter to run the model with.
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

    // Get number of elements in input tensor
    num_elements = model_input->bytes / sizeof(float);
    buf_len = sprintf(buf, "Number of input elements: %lu\r\n", num_elements);
    HAL_UART_Transmit(&huart1, (uint8_t *)buf, buf_len, 100);

    /* USER CODE END 2 */

    /* Infinite loop */
    /* USER CODE BEGIN WHILE */

    buf_len = sprintf(buf, "start!\n");
    HAL_UART_Transmit(&huart1, (uint8_t *)buf, buf_len, HAL_MAX_DELAY);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
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
//	buf_len = sprintf(buf, "1: %d|2: %d|3: %d|4: %d\r\n",
//				  seq2[0],seq2[1], seq2[2], seq2[3]);
//	HAL_UART_Transmit(&huart1, (uint8_t *)buf, buf_len, HAL_MAX_DELAY);

	int x = 0;   // Seq1 index
	int y = 0;   // Seq2 index

    //readUartBytes(model_input->data.f, imagesize);
	int total_actions = 0;
	int sub_actions = 0;
	// Get current timestamp

	error_reporter->Report("Starting sub-alignment...");

	while(x < l1 && y < l2)
	{
		// Get current timestamp
		timestamp = htim14.Instance->CNT;

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
		uint16_t timestamp_next = htim14.Instance->CNT;
		timestamp = (timestamp_next<timestamp)?(65536 - timestamp + timestamp_next): (timestamp_next-timestamp);

		total_time += (float)timestamp * 0.00038;
		//timestamp = htim14.Instance->CNT - timestamp;
		//----------
//		char str[250] = {0};
//		for (int i = 1; i < 4; i++)
//		{
//			sprintf(buf, "%e, ", static_cast<float>(model_output->data.f[i]));
//			strcat(str, buf);
//		}
//		strcat(str, "\n");
//
//		int len = strlen(str);
//		HAL_UART_Transmit(&huart1, (uint8_t *)str, len, HAL_MAX_DELAY);
		//----------
//		buf_len = sprintf(buf, "Duration: %lu\r\n", timestamp);
//		HAL_UART_Transmit(&huart1, (uint8_t *)buf, buf_len, 100);

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

	buf_len = sprintf(buf,
					  "Total actions: %lu | Duration(s): %.3f\r\n",
					  total_actions,
					  total_time);
	HAL_UART_Transmit(&huart1, (uint8_t *)buf, buf_len, 100);

	buf_len = sprintf(buf, "done!\n");
	HAL_UART_Transmit(&huart1, (uint8_t *)buf, buf_len, HAL_MAX_DELAY);

	//sendSeq(actions, total_actions);   // Send all actions
	//HAL_Delay(500);
	HAL_GPIO_TogglePin(GPIOI, GPIO_PIN_1);
    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

//void readUartBytes(float *data, int imageSize)
//{
//    uint8_t *rxBuffer = (uint8_t *)malloc(imageSize);
//    for(int i=0;i<imageSize;i++)
//    {
//    	HAL_UART_Receive(&huart1, (rxBuffer+i), 1, HAL_MAX_DELAY);
//    	data[i] = static_cast<float>(rxBuffer[i]) / 255.0f;
//    }
//    free(rxBuffer);
//    //rxBytes = HAL_UART_Receive(&huart1, rxBuffer, imageSize, HAL_MAX_DELAY);
//}

void readSeq(uint8_t *data, int length)
{
	//uint8_t *rxBuffer = (uint8_t *)malloc(length);
	for(int i=0; i<length; i++)
    {
    	HAL_UART_Receive(&huart1, (data + i), 1, HAL_MAX_DELAY);
    	//data[i] = rxBuffer[i];
    }
	//free(rxBuffer);
}

void sendSeq(uint8_t *data, int length)
{
	buf_len = sprintf(buf, "rc\n");
	HAL_UART_Transmit(&huart1, (uint8_t *)buf, buf_len, HAL_MAX_DELAY);

	buf_len = sprintf(buf, "%d\n", length);
	HAL_UART_Transmit(&huart1, (uint8_t *)buf, buf_len, HAL_MAX_DELAY);

	for(int i=0; i<length; i++)
	{
		//buf_len = sprintf(buf, "%d\n", data[i]);
		//HAL_UART_Transmit(&huart1, (uint8_t *)buf, buf_len, HAL_MAX_DELAY);
		HAL_UART_Transmit(&huart1, (data + i), 1, HAL_MAX_DELAY);
	}
}

float receive_float(void)
{
	union{
		float val_float;
		unsigned char bytes[4];
	}data;
	for(int i=0; i<4; i++)
		HAL_UART_Receive(&huart1, (uint8_t *)data.bytes + i, 1, HAL_MAX_DELAY);
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

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 100;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_USART1;
  PeriphClkInitStruct.Usart1ClockSelection = RCC_USART1CLKSOURCE_PCLK2;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM14 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM14_Init(void)
{

  /* USER CODE BEGIN TIM14_Init 0 */

  /* USER CODE END TIM14_Init 0 */

  /* USER CODE BEGIN TIM14_Init 1 */

  /* USER CODE END TIM14_Init 1 */
  htim14.Instance = TIM14;
  htim14.Init.Prescaler = 38000-1;
  htim14.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim14.Init.Period = 65535;
  htim14.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim14.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim14) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM14_Init 2 */

  /* USER CODE END TIM14_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOI_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOI, GPIO_PIN_1, GPIO_PIN_RESET);

  /*Configure GPIO pin : PI1 */
  GPIO_InitStruct.Pin = GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(GPIOI, &GPIO_InitStruct);
  HAL_GPIO_WritePin(GPIOI, GPIO_PIN_1, GPIO_PIN_RESET);
}

/* USER CODE BEGIN 4 */
extern "C" void DebugLog(const char* s)
{
	HAL_UART_Transmit(&huart1, (uint8_t *)s, strlen(s), 100);
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
