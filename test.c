
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

typedef unsigned int  uint32_t;
typedef unsigned long uint64_t;

typedef struct parts_t {
	uint32_t sign;
	uint32_t exp;
	uint32_t mantissa;
} parts_t;
 
parts_t float_to_parts(float val) {
  parts_t ret;
  uint32_t* ptr = (int*) &val;
  ret.sign     = ((*ptr) & 0x80000000) >> 31;
  ret.exp      = ((*ptr) & 0x7F800000) >> 23;
  ret.exp -= 127;
  ret.mantissa = ((*ptr) & 0x007FFFFF) >> 0;
  return ret;
}

float parts_to_float(parts_t parts) {
  uint32_t tmp;
  uint32_t exp = parts.exp + 127;
  tmp  = (parts.sign     & 0x00000001) << 31;
  tmp |= (exp            & 0x000000FF) << 23;
  tmp |= (parts.mantissa & 0x007FFFFF) << 0;
  float* ptr = (float*) &tmp;
  return (*ptr);
}

parts_t multiply(parts_t p1, parts_t p2) {
  parts_t ret;
  ret.sign = p1.sign * p2.sign;
  ret.exp = p1.exp + p2.exp;

  uint64_t prod = ((uint64_t) p1.mantissa) * ((uint64_t) p2.mantissa);
  prod = prod >> 23;

  uint32_t over = (p1.mantissa + p2.mantissa) >> 23;
  if (over == 0) {
    ret.mantissa = prod;
  }
  else if (over == 1) {
    ret.mantissa = prod >> 1;
    ret.exp = ret.exp + 1;
  }
  else {
    assert(0);
  }
  return ret;
}

int main() {
  float f1 = 1.5;
  parts_t p1 = float_to_parts(f1);
  printf("%d %d %d\n", p1.sign, p1.exp, p1.mantissa);

  parts_t p2 = multiply(p1, p1);
  float f2 = parts_to_float(p2);
  printf("%f\n", f2);  
}



