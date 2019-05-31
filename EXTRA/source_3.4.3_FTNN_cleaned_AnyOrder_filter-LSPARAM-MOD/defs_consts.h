#ifndef DEFS_CONSTS_H
#define DEFS_CONSTS_H

#define PI 3.141592653589793
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define SWAP(a,b) {register double tmp; tmp = a; a = b; b = tmp;}
#ifndef SQR
#define SQR(a) (a*a)
#endif
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define FILENAMESIZE 256
#define FERR __FILE__,__FUNCTION__,__LINE__
#define MAX_BOP_PARAM 9 // max. number of straight BOP parameters
#define MAX_LSP 7 // max. number of local structural parameters
#define MAX_HB_PARAM 8 // max. number of hybrid BOP parameters
//#define REF_GI 0.5

#endif // DEFS_CONSTS_H

