#include <math.h>
#include <stdlib.h>
#include <limits>
#include <sys/time.h>

#include "defs_consts.h"
#include "nrmatrix.h"
#include "util.h"
#include "nr.h"
#include "ran1.h"
#include "compute.h"

int ncom;        // Global variables communicate with f1dim.
VecDoub pcom, xicom;
double (*nrfunc)(VecDoub &);

double tt;
long int iseed;

/*double f1dim(double x)
// Must accompany linmin
{
  double f;
  VecDoub xt(ncom);
    
  for (int j=0; j<ncom; j++) xt[j] = pcom[j] + x * xicom[j];
  f = nrfunc(xt);

  return f;
}

void linmin (VecDoub &p, VecDoub &xi, int n, double &fret, double (&func)(VecDoub &))
// Line-minimization routine. Given an n-dimensional point p[0..n-1] and an n-dimensional
// direction xi[0..n-1], moves and resets p to where the function or functor func(p) takes on
// a minimum along the direction xi from p, and replaces xi by the actual vector displacement
// that p was moved. Also returns the value of func at the returned location p. This is actually
// all accomplished by calling the routines mnbrak and brent.
{
    double xx,xmin,fx,fb,fa,bx,ax;
    double TOL = 2.0e-4;
    
    ncom = n;
    pcom.resize (n);
    xicom.resize (n);
    nrfunc = func;
    for (int j=0; j<n; j++) {
        pcom[j] = p[j];
        xicom[j] = xi[j];
    }

    ax = 0.0; // Initial guess for brackets.
    xx = 1.0;
    mnbrak(ax,xx,bx,fa,fx,fb,f1dim);
    fret = brent(ax,xx,bx,f1dim,TOL,xmin);
    for (int j=0; j<n; j++) { // Construct the vector results to return
        xi[j] *= xmin;
        p[j] += xi[j];
    }
    //delete [] xicom;
    //delete [] pcom;
}


void mnbrak(double &ax, double &bx, double &cx, double &fa, double &fb,
            double &fc, double (&func)(double))
// Given a function or functor func, and given distinct initial points ax and bx, this routine
// searches in the downhill direction (defined by the function as evaluated at the initial points)
// and returns new points ax, bx, cx that bracket a minimum of the function. Also returned
// are the function values at the three points, fa, fb, and fc.
{
    // Here GOLD is the default ratio by which successive intervals are magnified and GLIMIT
    // is the maximum magnification allowed for a parabolic-fit step.
    double GOLD = 1.618034;
    double GLIMIT = 100.0;
    double TINY = 1.0e-20;
    double ulim,r,q,u,fu;
    
    fa = func(ax);
    fb = func(bx);
    if (fb > fa) {   // Switch roles of a and b so that we can go
        SWAP(ax,bx); // downhill in the direction from a to b.
        SWAP(fb,fa);
    }
    cx = bx + GOLD*(bx - ax); // First guess for c.
    fc = func(cx);
    while (fb > fc) {                     // Keep returning here until we bracket.
        r = (bx - ax) * (fb - fc);        // Compute u by parabolic extrapolation from
        q = (bx - cx) * (fb - fa);        // a,b,c. TINY is used to prevent any possible
        u = bx - ((bx-cx)*q - (bx-ax)*r)/ // division by zero.
                (2.0*SIGN(MAX(fabs(q-r),TINY),q-r));
        ulim = bx + GLIMIT*(cx - bx);
        // We don't go farther than this. Test various possibilities:
        if ((bx-u)*(u-cx) > 0.0) { // Parabolic u is between b and c: try it.
            fu = func(u);
            if (fu < fc) {        // Got a minimum between b anc c.
                ax = bx;
                bx = u;
                fa = fb;
                fb = fu;
                return;
            } else if (fu > fb) { // Got a minimum between a and u.
                cx = u;
                fc = fu;
                return;
            }
            u = cx + GOLD*(cx - bx); // Parabolic fit was no use. Use default magnification.
            fu = func(u);
        } else if ((cx-u)*(u-ulim) > 0.0) { // Parabolic fit is between c and
            fu = func(u);                   // its allowed limit.
            if (fu < fc) {
                shft3(bx,cx,u,cx+GOLD*(cx-bx));
                shft3(fb,fc,fu,func(u));
            }
        } else if ((u-ulim)*(ulim-cx) >= 0.0) { // Limit parabolic u to maximum allowed value
            u = ulim;
            fu = func(u);
        } else {                  // Reject parabolic u, use default magnification.
            u = cx + GOLD*(cx-bx);
            fu = func(u);
        }
        shft3(ax,bx,cx,u); // Eliminate oldest point and continue.
        shft3(fa,fb,fc,fu);
    }
}


double brent(double ax, double bx, double cx, double (&func)(double), double tol, double &xmin)
// Given a function or functor f, and given a bracketing triplet of abscissas ax, bx, cx (such
// that bx is between ax and cx, and f(bx) is less than both f(ax) and f(cx)), this routine
// isolates the minimum to a fractional precision of about tol using Brent's method. The
// abscissa of the minimum is returned as xmin, and the function value at the minimum is
// returned as min, the returned function value.
{
    int ITMAX = 500;
    double CGOLD = 0.3819660;
    const double ZEPS = std::numeric_limits<double>::epsilon()*1.0e-3;
    // Here ITMAX is the maximum allowed number of iterations; CGOLD is the golden ratio
    // and ZEPS is a small number that protects against trying to achieve fractional accuracy
    // for a minimum that happens to be exactly zero.
    double a,b,d=0.0,etemp,fu,fv,fw,fx;
    double p,q,r,tol1,tol2,u,v,w,x,xm;
    double e = 0.0;                    // This will be the distance moved on the step before last.
    a = (ax < cx ? ax : cx);           // a and b must be in ascending order,
    b = (ax > cx ? ax : cx);           // but input abscissas need not be.
    x = w = v = bx;                    // Initializations ...
    fw = fv = fx = func(x);
    for (int iter=0; iter<ITMAX; iter++) { // Main program loop.
        xm = 0.5*(a + b);
        tol2 = 2.0*(tol1 = tol*fabs(x) + ZEPS);
        if (fabs(x-xm) <= (tol2-0.5*(b-a))) { // Test for done here.
            xmin = x;
            return fx;
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
                d = CGOLD*(e = (x >= xm ? a-x : b-x));
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
        fu = func(u);
        // This is the one function evaluatiion per iteration.
        if (fu <= fx) {                      // Now decide what to do with our
            if (u >= x) a = x; else b = x;   // function evaluation.
            shft3(v,w,x,u);                   // Housekeeping follows:
            shft3(fv,fw,fx,fu);
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u;
                fv = fu;
            }
        }                               // Done with housekeeping. Back for
    }                                   // another iteratuion.
    errmsg("Too many iterations in brent",FERR);
    
    return 0;
}

void powell(VecDoub &p, MatDoub &xi, int n, double ftol, int &iter,
            double &fret, double (&func)(VecDoub &))
// Minimization of a function or functor of n variables. Input consists of an initial
// starting point p[0..n-1]. The initial matrix xi[0..n-1][0..n-1], whose columns
// contain the initial set of directions, is set to the n unit vectors. Returned
// is the best point found, at which point fret is the minimum function value and
// iter is the number of iterations taken. The routine linmin is used.
{
    const int ITMAX = iter; //200;
    const double TINY = 1.0e-25;
    double del,fp,fptt;
    int ibig;
    //double *pt,*ptt,*xit;
    VecDoub pt(n); // = new double [n];
    VecDoub ptt(n); // = new double [n];
    VecDoub xit(n); // = new double [n];

    fret = func(p);
    
    for (int j=0; j<n; j++) pt[j] = p[j];    // Save the initial point.
    for (iter=0;;++iter) {
        fp = fret;
        ibig = 0;
        del = 0.0;                // Will be the biggest function decrease.
        for (int i=0; i<n; i++) { // In each iteration, loop over all directions in the set.
            for (int j=0; j<n; j++) xit[j] = xi[j][i]; // Copy the direction,
            fptt = fret;
	    linmin(p,xit,n,fret,func);       // minimize along it,
	    if (fabs(fptt-fret) > del) {     // and record it if it is the largest decrease so far
		del = fabs(fptt - fret);
                ibig = i + 1;
            }
        }                                // Here comes the termination criterion:
	if (2.0*fabs(fp-fret) <= ftol*(fabs(fp) + fabs(fret)) + TINY) {
            return;                      // Termination criterion.
        }
        if (iter == ITMAX) {
            //errmsg("powell exceeding maximum iterations",FERR); XXXX
            return;
        }
        for (int j=0; j<n; j++) {        // Construct the extrapolated point and the
            ptt[j] = 2.0*p[j] - pt[j];   // average direction moved. Save the old starting point.
            xit[j] = p[j] - pt[j];
            pt[j] = p[j];
        }
        fptt = func(ptt);                // Function value at extrapolated point.
        if (fptt < fp) {
            double t = 2.0*(fp-2.0*fret+fptt)*SQR(fp-fret-del)-del*SQR(fp-fptt);
            if (t < 0.0) {
                linmin(p,xit,n,fret,func);  // Move to the minimum of the new direction,
                for (int j=0; j<n; j++) {   // and save the new direction.
                    xi[j][ibig-1] = xi[j][n-1];
		    //xi(j,ibig-1) = xi(j,n-1);
                    xi[j][n-1] = xit[j];
                    //xi(j,n-1) = xit[j];
                }
            }
        }
    }
}
*/

void dfpmin (VecDoub &p, const double gtol, int &iter,
             double &fret, double (&func)(VecDoub &))
// Given a starting point p[0..n-1], the Broyden-Fletcher-Goldfarb-Shanno variant of
// Davidson-Fletcher-Powell minimization is performed on a function whose value and
// gradient are provided by a functor df. The convergence requirement on zeroing
// the gradient is input as gtol. Returned quantities are p[0..n-1] (the location of
// the minimum), iter (the number of iterations that were performed), and fret (the
// minimum value of the function). The routine lnsrch is called to perform
// approximate line minimizations.
{
  const int ITMAX = iter; //200;
  const double EPS = std::numeric_limits<double>::epsilon();
  const double TOLX = 4*EPS, STPMX = 100.0; // STPMX = 100.0;
  // Here ITMAX is the maximum allowed number of iterations; EPS is the machine precision;
  // TOLX is the convergence criterion on x values; and STPMX is the scaled maximum step
  // length allowed in line searches.
  bool check;
  double den,fac,fad,fae,fp,stpmax,sum=0.0,sumdg,sumxi,temp,test;
  //double *dg,*g,*hdg,*pnew,*xi;
  int n = p.size ();
  MatDoub hessin(n,n);
  VecDoub dg(n); // = new double [n];
  VecDoub g(n); // = new double [n];
  VecDoub hdg(n); // = new double [n];
  VecDoub pnew(n); // = new double [n];
  VecDoub xi(n); // = new double [n];

  fp = func(p);                 // Calculate starting function value and
  df(p,g,func);                 // gradient,

  for (int i=0; i<n; i++) {     // and initialize the inverse Hessian to the
    for (int j=0; j<n; j++) hessin[i][j] = 0.0; // unit matrix.
    hessin[i][i] = 1.0;
    xi[i] = -g[i];           // Initial line direction.
    sum += p[i]*p[i];
  }
  stpmax = STPMX*MAX(sqrt(sum),double (n));
  for (int its=0; its<ITMAX; its++) { // Main loop over the iterations.
    iter = its;
    lnsrch(p,fp,g,xi,pnew,fret,stpmax,check,func);
    // The new function evaluation occurs in lnsrch; save the function value in fp for the
    // next line search. It is usually safe to ignore the value of check.
    fp = fret;
    for (int i=0; i<n; i++) {
      xi[i] = pnew[i] - p[i];   // Update the line direction,
      p[i] = pnew[i];           // and the current point.
    }
    test = 0.0;                   // Test for convergence on dx.
    for (int i=0; i<n; i++) {
      temp = fabs(xi[i])/MAX(fabs(p[i]),1.0);
      if (temp > test) test = temp;
    }
    if (test < TOLX) {
      return;
    }
    for (int i=0; i<n; i++) dg[i] = g[i];    // Save the old gradient,
    df(p,g,func);                            // and get the new gradient.
    test = 0.0;                              // Test for convergence on zero gradient.
    den = MAX(fret,1.0);
    for (int i=0; i<n; i++) {
      temp = fabs(g[i])*MAX(fabs(p[i]),1.0)/den;
      if (temp > test) test = temp;
    }
    if (test < gtol) {
      return;
    }
    for (int i=0; i<n; i++) dg[i] = g[i] - dg[i]; // Compute difference of gradients,
    for (int i=0; i<n; i++) {                     // and difference times current matrix.
      hdg[i] = 0.0;
      for (int j=0; j<n; j++) hdg[i] += hessin[i][j]*dg[j];
    }
    fac = fae = sumdg = sumxi = 0.0;   // Calculate dot products for the denominators.
    for (int i=0; i<n; i++) {
      fac += dg[i]*xi[i];
      fae += dg[i]*hdg[i];
      sumdg += pow(dg[i],2);
      sumxi += pow(xi[i],2);
    }
    if (fac > sqrt(EPS*sumdg*sumxi)) { // Skip update if fac not sufficiently positive.
      fac = 1.0/fac;
      fad = 1.0/fae;
      // The vector that makes BFGS different from DFP:
      for (int i=0; i<n; i++) dg[i] = fac*xi[i] - fad*hdg[i];
      for (int i=0; i<n; i++) {  // The BFGS updating formula:
	for (int j=0; j<n; j++) {
	  hessin[i][j] += fac*xi[i]*xi[j]
	      - fad*hdg[i]*hdg[j] + fae*dg[i]*dg[j];
	  //hessin(j,i) = hessin(i,j);
	}
      }
    }
    for (int i=0; i<n; i++) {  // Now calculate the next direction to go,
      xi[i] = 0.0;
      for (int j=0; j<n; j++) xi[i] -= hessin[i][j]*g[j];
    }
  }                             // and go back for another iteration.
  //errmsg ("too many iterations in dfpmin",FERR);
}

#include "derivatives.h"

void df(VecDoub &x, VecDoub &df, double (&func)(VecDoub &))
{
  //static int count = 0;

  // Set to approximate square root of the machine precision.
  double EPS = sqrt(std::numeric_limits<double>::epsilon());
  int n = x.size();
  VecDoub xh = x;
  double temp,h,fh;
  double fold;
  // This flag is passed to slaves processes
  // through func() to compute analytical derivatives.
  isLocalGrad = isGlobalGrad;

  //double stime = MPI_Wtime();
  fold = func(x);

  //
  // ----------------- Only for debugging purpose -----------------
  //printf("BOPconst=%e NNconst=%e HBconst=%e HBconst2=%e\n",
  //BOPconstraint,NNconstraint,HBconstraint,HBconstraint2);
  // ------------ end of debugging --------------------------------
  //

  //
  // ------------ Compute gradients ---------------------------------
  //
  if (!isLocalGrad) { // Compute derivatives using finite-difference.
    //printf("computing gradients using finite difference ...\n");
    for (int j=0; j<n; j++) {
      temp = xh[j];
      h = EPS*fabs(temp);
      if (h == 0.0) h = EPS;
      xh[j] = temp + h;          // Trick to reduce finite-precision error.
      h = xh[j] - temp;
      fh = func(xh);
      xh[j] = temp;
      df[j] = (fh - fold)/h;
      //printf("%d %.16e\n",j,df[j]);
    }
  } else {
    //
    // ------------------------------------------------------
    // Compute analytical derivatives:
    // Only master computes the derivatives for NN potential.
    // Must call func() first before calling the following subroutines !!!
    // ------------------------------------------------------
    //
    //printf("computing analytical derivatives ...\n");

    if (PotentialType == 1) computeNNDeriv(x, df);
    if (PotentialType == 2) computePINNDeriv(x, df);
  }
  // ----------- End of gradient calculations ------------------------
  //

  //
  // ------- only for debugging purpose ---------------
		/*count++;
  if (count == 5) {
    //for (int i=0; i<n; i++) std::cout << df[i] << "\n";
    int i = 0;
    // The first layer is the input layer and skipped.
    // Derivatives w.r.t weights.
    for (int l=1; l<nLayers; l++) {
      // weights: m x n matrix
      for (int m=0; m<layer[l-1].nnodes; m++) { // rows
	for (int n=0; n<layer[l].nnodes; n++) { // cols
	  printf("%d %e %d:[%d,%d]\n",i+1,df[i],l,m+1,n+1);
	  i++;
	}
      }
      // Derivatives w.r.t. biases.
      for (int m=0; m<layer[l].nnodes; m++) { // cols
	printf("%d %e %d:[1,%d]\n",i+1,df[i],l,m+1);
	i++;
      }
    }
    exit(0);
		}*/

  //printf("Elapsed time to compute finite/analytical derivatives: %e secs.\n",MPI_Wtime()-stime);
  // ------- end of last debugging -------------------------
  //

  // For the PINN potential, it tells Funk() to
  // compute only energies.
  isLocalGrad = 0;
}

void lnsrch(VecDoub &xold, const double fold, VecDoub &g, VecDoub &p,
            VecDoub &x, double &f, const double stpmax, bool &check,
            double (&func)(VecDoub &))
// Given an n-dimensional point xold[0..n-1], the value of the function and gradient there, fold
// and g[0..n-1], and a direction p[0..n-1], finds a new point x[0..n-1] along the direction
// p from xold where the function or functor func has decreased "sufficiently". The new function
// value is returned in f. stpmax is an input quantity that limits the length of the steps so that
// you do not try to evaluate the function in regions where it is undefined or subject to overflow.
// p is usually the Newton direction. The output quantity check is false on a normal exit. It is true
// when x is too close to xold. In a minimization algorithm, this usually signals convergence and
// can be ignored. However, in a zero-finding algorithm the calling program should check whether
// the convergence is spurious. Some "difficult" problems may require double precision in this routine.
{
  const double ALF = 1.e-4; //1.0e-4;
  const double TOLX = std::numeric_limits<double>::epsilon();
  // ALF ensures sufficient decrease in function value; TOLX is the convergence criterion on dx.
  double a,alam,alam2=0.0,alamin,b,disc,fold2,f2=0.0;
  double rhs1,rhs2,slope=0.0,sum=0.0,temp,test,tmplam;
  int i,n=xold.size();

  check=false;
  for (i=0; i<n; i++) sum += p[i]*p[i];
  sum = sqrt(sum);
  if (sum > stpmax) for (i=0; i<n; i++) p[i] *= stpmax/sum; // Scale if attempted step is too big.

  for (i=0; i<n; i++) slope += g[i]*p[i];
  if (slope >= 0.0) printf("Roundoff problem in lnsrch.\n"); //throw ("Roundoff problem in lnsrch.");
  test = 0.0;                                               // Compute lambda_min.
  for (i=0; i<n; i++) {
    temp = fabs(p[i])/MAX(fabs(xold[i]),1.0);
    if (temp > test) test = temp;
    }
  alamin = TOLX/test;
  alam = 1.0;                                             // Always try full Newton step first.

  for (;;) {                                              // Start of iteration loop.
    for (i=0; i<n; i++) x[i] = xold[i] + alam*p[i];   // Modified by Ganga P Purja Pun.
    f = func(x);
    if (alam < alamin) {                                // Convergence on dx. For zero finding,
      for (i=0; i<n; i++) x[i] = xold[i];             // the calling program should
      check = true;                                   // verify the convergence.
      return;
    } else if (f <= fold + ALF*alam*slope) return;      // Sufficient function decrease.
    else {                                              // Backtrack.
      if (alam == 1.0)
	tmplam = -slope/(2.0*(f-fold-slope));       // First time.
      else {                                          // Subsequent backtrack.
	rhs1 = f - fold - alam*slope;
	rhs2 = f2 - fold2 - alam2*slope;
	a = (rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2);
	b = (-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2);
	if (a == 0.0) tmplam = -slope/(2.0*b);
	else {
	  disc = b*b - 3.0*a*slope;
	  if (disc < 0.0) tmplam = 0.5*alam;
	  else if (b <= 0.0) tmplam = (-b + sqrt(disc))/(3.0*a);
	  else tmplam = -slope/(b + sqrt(disc));
	}
	if (tmplam > 0.5*alam) tmplam = 0.5*alam;
      }
    }
    alam2 = alam;
    f2 = f;
    fold2 = fold;
    alam = MAX(tmplam,0.1*alam);
  }  // Try again.
}

/*double golden (double ax, double bx, double cx, double (&func)(double),
               const double tol, double &xmin)
// Given a function func, and given a bracketing triplet of abscissas ax, bx, cx (such that bx is
// between ax and cx, and func(bx) is less than both func(ax) and func(cx), this routine performs a 
// golden section search for the minimum, isolating it to a fractional precision of about tol. The 
// abscissa of the minimum is returned as xmin, and the minimum function value is returned as 
// golden, the returned function value.
{
    const double R = 0.61803399;
    const double C = 1.0 - R;
    double f1,f2,x0,x1,x2,x3;
    
    x0 = ax;                         // At any given time we will keep track of four
    x3 = cx;                         // points, x0,x1,x2,x3.
    if (fabs(cx-bx) > fabs(bx-ax)) { // Make x0 to x1 the smaller segment,
        x1 = bx;
        x2 = bx + C*(cx - bx);       // and fill in the new point to be tried.
    } else {
        x2 = bx;
        x1 = bx - C*(bx - ax);
    }
    f1 = func(x1);                   // The initial function evaluations. Note that
    f2 = func(x2);                   // we never need to evaluate the function
    while (fabs(x3 - x0) > tol*(fabs(x1) + fabs(x2))) { // at the original endpoints.
        if (f2 < f1) {               // One possible outcome,
            shft3(x0,x1,x2,R*x1+C*x3); // its housekeeping,
            shft2(f1,f2,func(x2));     // and a new function evaluation.
        } else {                       // The other outcome,
            shft3(x3,x2,x1,R*x2+C*x0);
            shft2(f2,f1,func(x1));     // and its new function evaluation.
        }
    }                                  // Back to see if we are done.
    if (f1 < f2) {                     // We are done. Output the best of the two 
        xmin = x1;                     // current values.
        return f1;
    } else {
        xmin = x2;
        return f2;
    }
}

void chebft(const double a,const double b,double *c,const int n,double (*func)(double))
{
  int k,j;
  double fac,bpa,bma,*f,y,sum;

  f = new double [n];
  bma = 0.5*(b-a);
  bpa = 0.5*(b+a);
  for (k=0; k<n; k++) {
    y = cos(PI*(k+0.5)/n);
    f[k] = (*func)(y*bma+bpa);
		}
  fac = 2.0/n;
  for (j=0; j<n; j++) {
    sum = 0.0;
    for (k=0; k<n; k++) sum += f[k]*cos(PI*j*(k+0.5)/n);
    c[j] = fac*sum;
  }
  delete[] f;
}

double chebev(const double a,const double b,double *c,const int m,const double x)
{
  double d=0.0,dd=0.0,sv,y,y2;
  int j;

  if ((x-a)*(x-b)>0.0) {
    fprintf(stderr,"x not in range in routine chebev\n");
    exit(0);
  }
  y2 = 2.0*(y=(2.0*x-a-b)/(b-a));
  for (j=m-1; j>=1; j--) {
    sv = d;
    d = y2*d-dd+c[j];
    dd=sv;
  }
  return y*d-dd+0.5*c[0];
}

void chder(const double a,const double b,const double *c,double *cder,const int n)
{
  int j;
  double con;

  cder[n-1] = 0.0;
  cder[n-2] = 2*(n-1)*c[n-1];
  for (j=n-3; j>=0; j--) cder[j] = cder[j+2]+2*(j+1)*c[j+1];
  con = 2.0/(b-a);
  for (j=0; j<n; j++) cder[j] *= con;
}

void chint(const double a,const double b,const double *c,double *cint,const int n)
{
  int j;
  double sum=0.0,fac=1.0,con;

  con = 0.25*(b-a);
  for (j=1; j<=n-2;j++) {
    cint[j] = con*(c[j-1]-c[j+1])/j;
    sum += fac*cint[j];
    fac = -fac;
  }
  cint[n-1] = con*c[n-2]/(n-1);
  sum += fac*cint[n-1];
  cint[0] = 2.0*sum;
}

void amebsa(double **p, double *y, const int ndim, double *pb, double *yb,
            const double ftol, double (*funk)(double *), int *iter, double temptr)
// Multidimensional minimization of the function funk(x) where x[1..ndim] is a vector in
// ndim dimensions, by simulated annealing combined with the downhill simplex method of Nelder
// and Mead. The input matrix p[1..ndim+1][1..ndim] has ndim+1 rows, each an ndim-
// dimensional vector which is a vector of the starting simplex. Also input are the following: the
// vector y[1..ndim+1], whose components must be pre-initialized to the values of funk evaluated
// at the ndim+1 vertices (rows) of p; ftol, the fractional convergence tolerance to be achieved
// in the function value for an early return; iter, and temptr. The routine makes iter function
// evaluations at an annealing temperature temptr then returns. You should then decrease temptr
// according to your annealing schedule, reset iter, and call the routine again (leaving other
// arguments unaltered between calls). If iter is returned with a positive value, then early
// convergence and return occured. If you initialize yb to a very large value on the first
// call, then yb and pb[1..ndim] will subsequently return the best function value and point
// ever encountered (even if it is no longer a point in the simplex).
{
  double rtol,sum,swap,yhi,ylo,ynhi,ysave,yt,ytry,*psum;
  int i,ihi,ilo,j,m,n,mpts=ndim+1;

  psum = new double[ndim];

  tt = -temptr;
  for (n=0; n<ndim; n++) {
    for (sum=0.0,m=0; m<mpts; m++) sum += p[m][n];
    psum[n] = sum;
  }

  for (;;) {
    ilo = 0;                          // Determine which point is the highest (worst),
    ihi = 1;                          // next-highest, and lowest (best).
    ynhi = ylo = y[0]+tt*log(ran1(&iseed));  // Whenever we "look at" a vertex, it gets
    yhi = y[1]+tt*log(ran1(&iseed));         // a random thermal fluctuation.
    if (ylo > yhi) {
      ihi = 0;
      ilo = 1;
      ynhi = yhi;
      yhi = ylo;
      ylo = ynhi;
    }
    for (i=2; i<mpts; i++) { // Loop over the points in the simplex.
      yt = y[i]+tt*log(ran1(&iseed)); // More thermal fluctuations.
      if (yt <= ylo) {
        ilo = i;
        ylo = yt;
      }
      if (yt > yhi) {
        ynhi = yhi;
        ihi = i;
        yhi = yt;
      } else if (yt > ynhi) {
        ynhi = yt;
      }
    }
    rtol = 2.0*fabs(yhi-ylo)/(fabs(yhi)+fabs(ylo));
    // Compute the fractional range from highest to lowest and return if satisfactory.
    if (rtol < ftol || *iter < 0) { // If returning, put best point and value in
      swap = y[0];                  // slot 1.
      y[0] = y[ilo];
      y[ilo] = swap;
      for (n=0; n<ndim; n++) {
        swap = p[0][n];
        p[0][n] = p[ilo][n];
        p[ilo][n] = swap;
      }
      break;
    }
    *iter -= 2;
    // Begin a new iteration. First extrapolate by a factor -1 through the face of the simplex
    // across from the high point, i.e. reflect the simplex from the high point.
    ytry = amotsa(p,y,psum,ndim,pb,yb,funk,ihi,&yhi,-1.0);
    if (ytry <= ylo) {
      // Gives a result better than the best point, so try an additional extrapolation by a
      // factor of 2.
      ytry = amotsa(p,y,psum,ndim,pb,yb,funk,ihi,&yhi,2.0);
    } else if (ytry >= ynhi) {
      // The reflected point is worse than the second-highest, so look for an intermediate
      // lower point, i.e. do a one-dimensional contraction.
      ysave = yhi;
      ytry = amotsa(p,y,psum,ndim,pb,yb,funk,ihi,&yhi,0.5);
      if (ytry >= ysave) { // Can't seem to get rid of that high point.
        for (i=0; i<mpts; i++) { // Better contract around the lowest
          if (i != ilo) {        // (best) point.
            for (j=0; j<ndim; j++) {
              psum[j] = 0.5*(p[i][j]+p[ilo][j]);
              p[i][j] = psum[j];
            }
            y[i] = (*funk)(psum);
          }
        }
        *iter -= ndim;
        for (n=0; n<ndim; n++) { // Recompute psum.
          for (sum=0.0,m=0; m<mpts; m++) sum += p[m][n];
          psum[n] = sum;
        }
      }
    } else ++(*iter);
  }
  delete[] psum;
}


double amotsa(double **p, double *y, double *psum, const int ndim,
              double *pb, double *yb, double (*funk)(double *), int ihi,
              double *yhi, double fac)
// Extrapolates by a factor fac through the face of the simplex across from the
// high point, tries it, and replaces the high point if the new point is better.
{
  int j;
  double fac1,fac2,yflu,ytry,*ptry;

  ptry = new double[ndim];
  fac1 = (1.0-fac)/ndim;
  fac2 = fac1 - fac;
  for (j=0; j<ndim; j++) ptry[j] = psum[j]*fac1-p[ihi][j]*fac2;
  ytry = (*funk)(ptry);
  if (ytry <= *yb) {
    for (j=0; j<ndim; j++) pb[j] = ptry[j];
    *yb = ytry;
  }
  // tt is a global variable which holds negative of the scheduling temperature
  yflu = ytry-tt*log(ran1(&iseed)); // We added a thermal fluctuation to all the current
  if (yflu < *yhi) {         // vertices, but we subtract it here, so as to give
    y[ihi] = ytry;           // the simplex a thermal Brownian motion: It
    *yhi = yflu;             // likes to accept any suggested change.
    for (j=0; j<ndim; j++) {
      psum[j] += ptry[j]-p[ihi][j];
      p[ihi][j] = ptry[j];
    }
  }
  delete[] ptry;
  return yflu;
}
*/

/*void steepest_descent(VecDoub &p, const double gamma, int &iter,
		      double &fret, double (&func)(VecDoub &))
// Given a starting point p[0..n-1], the steepest descent is performed on a
// function whose value and gradient are provided by a functor df.
// The step size is input as gamma. Returned quantities are p[0..n-1]
// (the location of the minimum), iter (the number of iterations that
// were performed), and fret (the minimum value of the function).
{
  const int ITMAX = iter;
  const double EPS = std::numeric_limits<double>::epsilon();
  // Here ITMAX is the maximum allowed number of iterations; EPS is the machine precision;

  int n = p.size ();
  double fp;
  VecDoub g(n); // = new double [n];

  fp = func(p); // Calculate starting function value and gradient.
  df(p,g,func);

  for (int its=0; its<ITMAX; its++) { // Main loop over the iterations.
    iter = its + 1;
    for (int i=0; i<n; i++) p[i] += -gamma*g[i];
    fret = func(p);
    if (fabs(fp - fret) < EPS) return;
    fp = fret;
    df(p,g,func);
  }
}


void steepest_descent2(VecDoub &p, double &gamma, int &iter,
		       double &fret, double (&func)(VecDoub &))
// Given a starting point p[0..n-1], the steepest descent with variable step size
// is performed on a function whose value and gradient are provided by a functor df.
// The initial step size is input as gamma. Returned quantities are p[0..n-1]
// (the location of the minimum), iter (the number of iterations that were
// performed), and fret (the minimum value of the function).
{
  const int ITMAX = iter;
  const double EPS = std::numeric_limits<double>::epsilon();
  // Here ITMAX is the maximum allowed number of iterations; EPS is the machine precision;

  int n = p.size ();
  double fp;
  VecDoub x = p;
  VecDoub prev_g(n);
  VecDoub g(n);

  fp = func(p); // Calculate starting function value and gradient.
  df(p,prev_g,func);

  for (int its=0; its<ITMAX; its++) { // Main loop over the iterations.
    iter = its + 1;
    for (int i=0; i<n; i++) p[i] += -gamma*prev_g[i];
    fret = func(p);
    if (fabs(fp - fret) < EPS) return;
    fp = fret;
    df(p,g,func);  // Compute new gradient based on updated vector.
    double num, denum;
    num = denum = 0.0;
    for (int i=0; i<n; i++) {
      denum += pow(g[i] - prev_g[i],2);
      num += (p[i] - x[i])*(g[i] - prev_g[i]);
    }
    gamma = num/denum;  // Find new gamma.
    x = p;  // Update previous vector as new.
    prev_g = g;  // Update previous gradient as new.
  }
}


void steepest_descent3(VecDoub &p, VecDoub &gamma, int &iter,
		       double &fret, double (&func)(VecDoub &))
// Given a starting point p[0..n-1], the steepest descent with variable step size
// is performed on a function whose value and gradient are provided by a functor df.
// The initial step size is input as gamma. Returned quantities are p[0..n-1]
// (the location of the minimum), iter (the number of iterations that were
// performed), and fret (the minimum value of the function).
{
  const int ITMAX = iter;
  const double EPS = std::numeric_limits<double>::epsilon();
  // Here ITMAX is the maximum allowed number of iterations; EPS is the machine precision;

  int n = p.size ();
  double fp;
  VecDoub x = p;
  VecDoub prev_g(n);
  VecDoub g(n);

  fp = func(p); // Calculate starting function value and gradient.
  df(p,prev_g,func);

  for (int its=0; its<ITMAX; its++) { // Main loop over the iterations.
    iter = its + 1;
    for (int i=0; i<n; i++) p[i] += -gamma[i]*prev_g[i];
    fret = func(p);
    if (fabs(fp - fret) < EPS) return;
    fp = fret;
    df(p,g,func);  // Compute new gradient based on updated vector.
    for (int i=0; i<n; i++) {
      double denum = pow(g[i] - prev_g[i],2);
      double num = (p[i] - x[i])*(g[i] - prev_g[i]);
      gamma[i] = num/denum;
    }
    x = p;  // Update previous vector as new.
    prev_g = g;  // Update previous gradient as new.
  }
}
*/
