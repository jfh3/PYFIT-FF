//#include "nrmatrix.h"
//#include "nrvector.h"

struct Amoeba {
  const double ftol;
  const int NMAX;
  int nfunc;
  int mpts;
  int ndim;
  double fmin;
  VecDoub y;
  MatDoub p;
  Amoeba(const double ftoll,const int maxnfunc) : ftol(ftoll),NMAX(maxnfunc) {} // XXXX
  template <class T>
  VecDoub minimize(VecDoub_I &point, const double del, T &func)
  {
    VecDoub dels(point.size(),del);
    return minimize(point,dels,func);
  }
  template <class T>
  VecDoub minimize(VecDoub_I &point, VecDoub_I &dels, T &func)
  {
    int ndim=point.size();
    MatDoub pp(ndim+1,ndim);
    for (int i=0;i<ndim+1;i++) {
      for (int j=0;j<ndim;j++)
	pp[i][j]=point[j];
      //pp(i,j) = point[j];
      if (i !=0 ) pp[i][i-1] += dels[i-1];
    }
    return minimize(pp,func);
  }
  template <class T>
  VecDoub minimize(MatDoub &pp, T &func)
  {
    //const int NMAX=5000; XXXX
    const double TINY=1.0e-10;
    int ihi,ilo,inhi;
    mpts=pp.nrows();
    ndim=pp.ncols();
    VecDoub psum(ndim),pmin(ndim),x(ndim);
    p = pp;
    y.resize(mpts);
    for (int i=0;i<mpts;i++) {
      for (int j=0;j<ndim;j++)
	x[j] = p[i][j];
      y[i]=func(x);
    }
    nfunc=0;
    get_psum(p,psum);
    for (;;) {
      ilo = 0;
      ihi = y[0]>y[1] ? (inhi=1,0) : (inhi=0,1);
      for (int i=0;i<mpts;i++) {
	if (y[i] <= y[ilo]) ilo=i;
	if (y[i] > y[ihi]) {
	  inhi=ihi;
	  ihi=i;
	} else if (y[i] > y[inhi] && i != ihi) inhi=i;
      }
      double rtol=2.0*fabs(y[ihi]-y[ilo])/(fabs(y[ihi])+fabs(y[ilo])+TINY);
      if (rtol < ftol || nfunc >= NMAX) {
	SWAP(y[0],y[ilo]);
	for (int i=0;i<ndim;i++) {
	  SWAP(p[0][i],p[ilo][i]);
	  //SWAP(p(0,i),p(ilo,i));
	  pmin[i] = p[0][i];
	}
	fmin=y[0];
	return pmin;
      }
      if (nfunc >= NMAX) { // commented by Ganga
	//throw("NMAX exceeded");
	fmin = y[0];
	return pmin;
      }
      nfunc += 2;
      double ytry=amotry(p,y,psum,ihi,-1.0,func);
      if (ytry <= y[ilo])
	ytry=amotry(p,y,psum,ihi,2.0,func);
      else if (ytry >= y[inhi]) {
	double ysave=y[ihi];
	ytry=amotry(p,y,psum,ihi,0.5,func);
	if (ytry >= ysave) {
	  for (int i=0;i<mpts;i++) {
	    if (i != ilo) {
	      for (int j=0;j<ndim;j++)
		p[i][j]=psum[j]=0.5*(p[i][j]+p[ilo][j]);
		//p(i,j) = psum[j] = 0.5*(p(i,j) + p(ilo,j));
	      y[i] = func(psum);
	    }
	  }
	  nfunc += ndim;
	  get_psum(p,psum);
	}
      } else --nfunc;
    }
  }
  inline void get_psum(MatDoub &p, VecDoub_O &psum)
  {
    for (int j=0;j<ndim;j++) {
      double sum=0.0;
      for (int i=0;i<mpts;i++)
	sum += p[i][j];
      psum[j]=sum;
    }
  }
  template <class T>
  double amotry(MatDoub &p, VecDoub_O &y, VecDoub_IO &psum,
		const int ihi, const double fac, T &func)
  {
    VecDoub ptry(ndim);
    double fac1=(1.0-fac)/ndim;
    double fac2=fac1-fac;
    for (int j=0;j<ndim;j++)
      ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
    //ptry[j] = psum[j]*fac1 - p(ihi,j)*fac2;
    double ytry=func(ptry);
    if (ytry < y[ihi]) {
      y[ihi]=ytry;
      for (int j=0;j<ndim;j++) {
	psum[j] += ptry[j] - p[ihi][j];
	p[ihi][j] = ptry[j];
	//p(ihi,j) = ptry[j];
      }
    }
    return ytry;
  }
};
