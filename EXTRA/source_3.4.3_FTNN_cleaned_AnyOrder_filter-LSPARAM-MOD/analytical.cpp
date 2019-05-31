#include <iostream>
#include <math.h> // cmath
#include <cstdlib>

#include "globals.h"
#include "analytical.h"
#include "nrmatrix.h"
#include "write.h"
#include "NNetInterface.h"

// ------- Heaviside step function ----------------
double HeavisideStepFunc(const double x)
{
  if (x<0) return 0;
  else return 1.0;
}


// ------- Cutoff function ------------------
void CutoffFunc(const double r, const double rc, const double hc,
                double &c0, double &c1, double &c2, const int m)
{
  double x,y,x4;

  x = r - rc;
  x = x/hc;
  x4 = pow(x,4);
  y = 1.0 + x4;

  c0 = c1 = c2 = 0.0;
  if (m==0) c0 = x4/y;
  else {
    c0 = x4/y;
    double x1 = 4.0*x*x;
    double y2 = y*y;
    c1 = x1*x/y2/hc;
    c2 = x1*(3.0 - 5.0*x4)/y2/y/hc/hc;
  }
}

double CutoffFunc(const double r, const double rc, const double hc)
{
  double x,y,x4;

  if (r > rc) return 0.0;
  
  x = r - rc;
  x = x/hc;
  x4 = pow(x, 4);
  y = 1.0 + x4;

  return x4/y;
}
