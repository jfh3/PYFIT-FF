#ifndef UTIL_H
#define UTIL_H

#include "nrmatrix.h"
#include "nrvector.h"

void errmsg(const char *msg, const char *file, const char *func, const int line);

inline void shft2(double &a, double &b, const double c)
{
    a = b;
    b = c;
}

inline void shft3(double &a, double &b, double &c, const double d)
{
    a = b;
    b = c;
    c = d;
}

inline void mov3(double &a, double &b, double &c, const double d,
                 const double e, const double f)
{
    a = d; b = e; c = f;
}

bool search_file(const char *fname);
int compareAbsMaxArray(const int n, const double *x, const double y, double &xout);
int compareAbsMaxVec(const VecDoub &x, const double y, double &xout);

#endif // UTIL_H
