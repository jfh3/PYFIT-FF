//#include "nrmatrix.h"
//#include "nrvector.h"
#include <limits>
#include "ran.h"

// Multidimensional minimization of the function funk(point) where point[1..ndim] is a vector in
// ndim dimensions, by simulated annealing combined with the downhill simplex method of Nelder
// and Mead. The input matrix p[1..ndim+1][1..ndim] has ndim+1 rows, each an ndim-
// dimensional vector which is a vector of the starting simplex. Also input are the following: the
// vector y[1..ndim+1], whose components must be pre-initialized to the values of funk evaluated
// at the ndim+1 vertices (rows) of p; ftol, the fractional convergence tolerance to be achieved
// in the function value for an early return; iter, and temptr. The routine makes iter function
// evaluations at an annealing temperature temptr then returns. You should then decrease temptr
// according to your annealing schedule, reset iter, and call the routine again (leaving other
// arguments unaltered between calls). If anneal returns true, then early convergence reached else
// it ran out of given iterations. If you initialize yb to a very large value on the first
// call, then yb and pb[1..ndim] will subsequently return the best function value and point
// ever encountered (even if it is no longer a point in the simplex).

//template <class T>
struct Amebsa {
  //T &funk;
  double (*funk)(VecDoub &);
  const double ftol;
  Ranq1 ran;
  double yb;
  int ndim;
  VecDoub pb;
  int mpts;
  VecDoub y;
  MatDoub p; // this Matrix class does not have [][] operator defined !!! instead (i,j) works.
  double tt;

  Amebsa(VecDoub_I &point, const double del, double (&funkk)(VecDoub &), const double ftoll) :
    funk(funkk), ftol(ftoll), ran(1234),
    yb(std::numeric_limits<double>::max()), ndim(point.size()), pb(ndim),
    mpts(ndim+1), y(mpts), p(mpts,ndim)
  {
    for (int i=0;i<mpts;i++) {
      for (int j=0;j<ndim;j++)
	p[i][j]=point[j];
      if (i != 0) p[i][i-1] += del;
    }
    inity();
  }

  Amebsa(VecDoub_I &point, VecDoub_I &dels, double (&funkk)(VecDoub &), const double ftoll) :
    funk(funkk), ftol(ftoll), ran(time(NULL)),
    yb(std::numeric_limits<double>::max()), ndim(point.size()), pb(ndim),
    mpts(ndim+1), y(mpts), p(mpts,ndim)
  {
    for (int i=0;i<mpts;i++) {
      for (int j=0;j<ndim;j++)
	p[i][j]=point[j];
      if (i != 0) p[i][i-1] += dels[i-1];
    }
    inity();
  }

  Amebsa(MatDoub &pp, double (&funkk)(VecDoub &), const double ftoll) : funk(funkk),
    ftol(ftoll), ran(1234), yb(std::numeric_limits<double>::max()),
    ndim(pp.ncols()), pb(ndim), mpts(pp.nrows()), y(mpts), p(pp)
  { inity(); }

  void inity() {
    VecDoub x(ndim);
    for (int i=0;i<mpts;i++) {
      for (int j=0;j<ndim;j++)
	x[j]=p[i][j];
      y[i]=funk(x);
    }
  }

  bool anneal(int &iter, const double temperature)
  {
    VecDoub psum(ndim);
    tt = -temperature;
    get_psum(p,psum);
    for (;;) {
      int ilo=0;                           // Determine which point is the highest (worst),
      int ihi=1;                           // next-highest, and lowest (best).
      double ylo=y[0]+tt*log(ran.doub());  // Whenever we "look at" a vertex, it gets
      double ynhi=ylo;                     // a random thermal fluctuation.
      double yhi=y[1]+tt*log(ran.doub());
      if (ylo > yhi) {
	ihi=0;
	ilo=1;
	ynhi=yhi;
	yhi=ylo;
	ylo=ynhi;
      }
      for (int i=3;i<=mpts;i++) {            // Loop over the points in the simplex.
	double yt=y[i-1]+tt*log(ran.doub()); // More thermal fluctuations.
	if (yt <= ylo) {
	  ilo=i-1;
	  ylo=yt;
	}
	if (yt > yhi) {
	  ynhi=yhi;
	  ihi=i-1;
	  yhi=yt;
	} else if (yt > ynhi) {
	  ynhi=yt;
	}
      }
      double rtol=2.0*fabs(yhi-ylo)/(fabs(yhi)+fabs(ylo));
      // Compute the fractional range from highest to lowest and return if satisfactory.
      if (rtol < ftol || iter < 0) {  // If returning, put best point and value in slot 0.
	SWAP(y[0],y[ilo]);
	for (int n=0;n<ndim;n++)
	  SWAP(p[0][n],p[ilo][n]);
	if (rtol < ftol)
	  return true;
	else
	  return false;
      }
      iter -= 2;
      // Begin a new iteration. First extrapolate by a factor -1 through the face of the simplex
      // across from the high point, i.e., reflect the simplex from the high point.
      double ytry=amotsa(p,y,psum,ihi,yhi,-1.0);
      if (ytry <= ylo) {
	// Gives a result better than the best point, so try an additional extrapolation by a
	// factor of 2.
	ytry=amotsa(p,y,psum,ihi,yhi,2.0);
      } else if (ytry >= ynhi) {
	// The reflected point is worse than the second-highest, so look for an intermediate
	// lower point, i.e., do a one-dimensional contraction.
	double ysave=yhi;
	ytry=amotsa(p,y,psum,ihi,yhi,0.5);
	if (ytry >= ysave) {          // Can't seem to get rid of that high point.
	  for (int i=0;i<mpts;i++) {  // Better contract around the lowest
	    if (i != ilo) {           // (best) point.
	      for (int j=0;j<ndim;j++) {
		psum[j]=0.5*(p[i][j]+p[ilo][j]);
		p[i][j]=psum[j];
	      }
	      y[i]=funk(psum);
	    }
	  }
	  iter -= ndim;
	  get_psum(p,psum);  // Recompute psum.
	}
      } else ++iter;         // Correct the evaluation count.
    }
  }

  inline void get_psum(MatDoub &p, VecDoub_O &psum)
  {
    for (int n=0;n<ndim;n++) {
      double sum=0.0;
      for (int m=0;m<mpts;m++) sum += p[m][n];
      psum[n]=sum;
    }
  }

  double amotsa(MatDoub &p, VecDoub_O &y, VecDoub_IO &psum,
	      const int ihi, double &yhi, const double fac)
  {
    VecDoub ptry(ndim);
    double fac1=(1.0-fac)/ndim;
    double fac2=fac1-fac;
    for (int j=0;j<ndim;j++)
      ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
    double ytry=funk(ptry);
    if (ytry <= yb) {
      for (int j=0;j<ndim;j++) pb[j]=ptry[j];
      yb=ytry;
    }
    double yflu=ytry-tt*log(ran.doub());  // We added a thermal fluctuation to all the current
    if (yflu < yhi) {                     // vertices, but we subtract it here, so as to give
      y[ihi]=ytry;                        // the simplex a thermal Brownian motion: It
      yhi=yflu;                           // likes to accept any suggested change.
      for (int j=0;j<ndim;j++) {
	psum[j] += ptry[j]-p[ihi][j];
	p[ihi][j]=ptry[j];
      }
    }
    return yflu;
  }
};
