#ifndef NR_H
#define NR_H

#include "nrvector.h"

extern int ncom;        // Defined in linmin.
extern VecDoub pcom, xicom;
extern double (*nrfunc)(VecDoub &);

extern double tt; // used in amebsa() routine.
extern long int iseed;

/*double f1dim(double x);

void chebft(const double a,const double b,double *c,const int n,double (*func)(double));

double chebev(const double a,const double b,const double *c,const int m,const double x);

void chder(const double a,const double b,const double *c,double *cder,const int n);

void chint(const double a,const double b,const double *c,double *cint,const int n);

void linmin(VecDoub &p, VecDoub &xi, int n, double &fret, double (&func)(VecDoub &));

void linmin(VecDoub &p, VecDoub &xi, int n, double &fret);

void mnbrak(double &ax, double &bx, double &cx, double &fa, double &fb,
	    double &fc, double (&func)(double));

double brent(double ax, double bx, double cx, double (&f)(double), double tol, double &xmin);

double golden (double ax, double bx, double cx, double (&func)(double), 
               const double tol, double &xmin);

void powell(VecDoub &p, MatDoub &xi, int n, double ftol, int &iter,
            double &fret, double (&func)(VecDoub &));
*/
void dfpmin(VecDoub &p, const double gtol, int &iter,
             double &fret, double (&func)(VecDoub &));

/*void steepest_descent(VecDoub &p, const double gamma, int &iter,
	     double &fret, double (&func)(VecDoub &));

void steepest_descent2(VecDoub &p, double &gamma, int &iter,
		       double &fret, double (&func)(VecDoub &));

void steepest_descent3(VecDoub &p, VecDoub &gamma, int &iter,
		       double &fret, double (&func)(VecDoub &));
*/

void lnsrch(VecDoub &xold, const double fold, VecDoub &g, VecDoub &p,
            VecDoub &x, double &f, const double stpmax, bool &check, double (&func)(VecDoub &));

void df(VecDoub &x, VecDoub &df, double (&func)(VecDoub &));

/*double amotsa(double **p, double *y, double *psum, const int ndim,
              double *pb, double *yb, double (*funk)(double *), int ihi,
              double *yhi, double fac);

void amebsa(double **p, double *y, const int ndim, double *pb, double *yb,
            const double ftol, double (*funk)(double *), int *iter, double temptr);
*/
#endif // NR_H

