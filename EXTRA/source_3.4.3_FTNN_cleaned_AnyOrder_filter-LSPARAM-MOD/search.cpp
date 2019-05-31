#include <math.h> // cmath

#include "search.h"

double fmin(const double ax, const double bx,
            double (*f)(const double), const double tol)
{
  /* =================================================================
  c  A slightly modified version of the 1D-minimization code from:
  c
  c     "Computer Methods for Mathematical Computations"
  c       by G.E. Forsythe, M.A. Malcolm and C.B. Moler
  c        (Englewood Cliffs, NJ: Prentice-Hall, 1977)
  c
  c  Retrieves the best estimate (with accuracy tol) of the point
  c  within the interval (ax,bx) where function F(x) attains a
  c  minimum
  c================================================================= */

  double a,b,d,etemp,fu,fv,fw,fx,p,q,r,eps,tol1,tol2,u,v,w,x,xm;
  double e=0.0;             // This will be the distance moved on
                            // the step before last.
  // eps is approximately the square root of the relative machine precision.
  eps = 1.0;
  do {
    eps = eps/2.0;
    tol1 = 1.0 + eps;
  } while (tol1 > 1.0);
  eps = sqrt(eps);

  a = (ax < bx ? ax : bx);                // a and b must be in ascending order,
  b = (ax > bx ? ax : bx);                // but input abscissas need not be.
  x=w=v=bx;                               // Initializations ...
  fw=fv=fx=(*f)(x);
  for (;;) {                              // Main program loop.
    xm = 0.5*(a+b);
    tol2 = 2.0*(tol1=tol*fabs(x)+eps);
    if (isnan(x)) return x;
    if (fabs(x-xm) <= (tol2-0.5*(b-a))) { // Test for done here.
      // return fx;                       // The minimum function value is not returned.
      return x;
    }
    if (fabs(e) > tol1) {               // Construct a trial parabolic fit.
      r = (x-w)*(fx-fv);
      q = (x-v)*(fx-fw);
      p = (x-v)*q-(x-w)*r;
      q = 2.0*(q-r);
      if (q > 0.0) p = -p;
      q = fabs(q);
      etemp = e;
      e = d;
      if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x)) {
        d = CGOLD*(e=(x >= xm ? a-x : b-x));
      }
      // The above conditions determine the acceptibility of the parabolic fit. Here we
      // take the golden section step into the larger of the two segments.
      else {
        d = p/q;                       // Take the parabolic step.
        u = x+d;
        if (u-a < tol2 || b-u < tol2) d = SIGN(tol1,xm-x);
      }
    } else {
      d = CGOLD*(e=(x >= xm ? a-x : b-x));
    }
    u = (fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
    fu = (*f)(u);
    // This is the one function evaluatiion per iteration.
    if (fu <= fx) {                    // Now decide what to do with our
      if (u >= x) a = x; else b = x;   // function evaluation.
      SHFT(v,w,x,u);                   // Housekeeping follows:
      SHFT(fv,fw,fx,fu);
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w;
        w = u;
        fv = fw;
        fw = fu;
      } else if (fu <= fv || v==x || v==w) {
        v = u;
        fv = fu;
      }
    }                               // Done with housekeeping. Back for
  }                                 // another iteration.
}
