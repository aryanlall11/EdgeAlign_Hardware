#ifndef PERIPHERAL_H /* prevent circular inclusions */
#define PERIPHERAL_H /* by using protection macros */
//#include "xfullyconnected.h"
#include "xconv_50x2.h"
#include "xconv_100x4.h"
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif
//extern XFullyconnected xfc;
//extern XFullyconnected_Config *xfc_cfg;

extern XConv_50x2 xconv;
extern XConv_50x2_Config *xconv_cfg;

extern XConv_100x4 xconvL;
extern XConv_100x4_Config *xconvL_cfg;

int InitPeripherals( void );

#ifdef __cplusplus
}
#endif

#endif
