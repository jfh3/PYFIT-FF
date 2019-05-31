#ifndef SEARCH_H
#define SEARCH_H

#define CGOLD (0.5*(3.0-sqrt(5.0)))
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
#ifndef SIGN
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#endif

double fmin(const double ax, const double bx,
            double (*f)(const double), const double tol);

#endif // SEARCH_H
