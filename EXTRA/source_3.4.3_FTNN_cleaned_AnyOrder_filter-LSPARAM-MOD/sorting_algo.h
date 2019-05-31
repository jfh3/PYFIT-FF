#ifndef SORTING_ALGO_H
#define SORTING_ALGO_H

#include <vector>

std::vector<int>::iterator
locate(const double x, const double *p, std::vector<int>::iterator il,
							std::vector<int>::iterator ih, int dir);

void NewSort(double *x,double **y,const int n,const int m,const int dir);

void insert(double *x, double **y, const int n, const int m,
            const double xi, const double *yi, const int n1, const int dir);

#endif // SORTING_ALGO_H
