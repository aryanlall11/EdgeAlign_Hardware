//* --------Included Headers--------
#include "xparameters.h"
#include "peripheral.h"

// Instance of the UART, local to this module
//XFullyconnected xfc;
//XFullyconnected_Config *xfc_cfg;

XConv_50x2 xconv;
XConv_50x2_Config *xconv_cfg;

XConv_100x4 xconvL;
XConv_100x4_Config *xconvL_cfg;

int InitPeripherals( void )
{
    int status;

//    print("Initializing Fully connected module...\n");
//    xfc_cfg = XFullyconnected_LookupConfig(XPAR_FULLYCONNECTED_0_DEVICE_ID);
//    if(xfc_cfg)
//    {
//        status = XFullyconnected_CfgInitialize(&xfc, xfc_cfg);
//        if(status != XST_SUCCESS){
//            print("Error Initializing FC module!\n");
//            return XST_FAILURE;}
//    }

    print("Initializing Convolution_50x2 module...\n");
	xconv_cfg = XConv_50x2_LookupConfig(XPAR_CONV_50X2_0_DEVICE_ID);
	if(xconv_cfg)
	{
		status = XConv_50x2_CfgInitialize(&xconv, xconv_cfg);
		if(status != XST_SUCCESS){
			print("Error Initializing CNN_50x2 module!\n");
			return XST_FAILURE;}
	}

	print("Initializing Convolution_100x4 module...\n");
	xconvL_cfg = XConv_100x4_LookupConfig(XPAR_CONV_100X4_0_DEVICE_ID);
	if(xconvL_cfg)
	{
		status = XConv_100x4_CfgInitialize(&xconvL, xconvL_cfg);
		if(status != XST_SUCCESS){
			print("Error Initializing CNN_50x2 module!\n");
			return XST_FAILURE;}
	}

    return XST_SUCCESS;
}
