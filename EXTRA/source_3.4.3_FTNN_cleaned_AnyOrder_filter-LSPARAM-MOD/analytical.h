#ifndef ANALYTICAL_H
#define ANALYTICAL_H

//double Funk(double *pin);

double HeavisideStepFunc(const double x);

void CutoffFunc(const double r, const double rc, const double hc,
		double &c0, double &c1, double &c2, const int m);

double CutoffFunc(const double r, const double rc, const double hc);

#endif // ANALYTICAL_H
