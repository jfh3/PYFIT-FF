#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "sorting_algo.h"
#include "mem.h"

void NewSort(double *x,double **y,const int n,const int m,const int dir)
/* x is a 1D array of size n; \
 * y is a 2D array of size n x m;
 */
{
  std::vector<int> myvector;
  std::vector<int>::iterator iam;
  double *tmpX,**tmpY;

  tmpX = create(tmpX,n,"NewSort: tmpX");
  tmpY = create(tmpY,n,m,"NewSort: tmpY");

  myvector.push_back(0);

  for (int i=1; i<n; i++) {
      iam = locate(x[i],x,myvector.begin(),myvector.end(),dir);
    if (dir == 1) {
      if (x[i]>x[*iam]) myvector.insert(iam+1,i);
      else myvector.insert(iam,i);
    }
    if (dir == -1) {
      if (x[i]<x[*iam]) myvector.insert(iam+1,i);
      else myvector.insert(iam,i);
    }
  }

  // Copy x and y to temporary variables in a given order;
  // Always do member by member copying for multidimensional arrays !!!; 
  iam=myvector.begin();
  for (int i=0; i<n && iam<myvector.end(); i++) {
    tmpX[i] = x[*iam];
    for (int j=0; j<m; j++) tmpY[i][j] = y[*iam][j];
    iam++;
  }
  // Copy back the ordered data to x and y;
  // Always do member by member copying for multidimensional arrays !!!;
  for (int i=0; i<n; i++) {
    x[i] = tmpX[i];
    for (int j=0; j<m; j++) y[i][j] = tmpY[i][j];
  }

  sfree(tmpX);
  destroy(tmpY);
  //for (iam=myvector.begin(); iam<myvector.end(); iam++) std::cout << *iam << " " << d[*iam] << "\n";
}

std::vector<int>::iterator
locate(const double x,const double *p,std::vector<int>::iterator il,
       std::vector<int>::iterator ih,int dir)
{
  unsigned int jump;
  ldiv_t divresult;
  std::vector<int>::iterator itr;

  if ((ih-il) == 1) return il;
  else {
    do {
      divresult = ldiv(ih-il,2L);
      jump = divresult.quot + divresult.rem;
      //std::cout << divresult.quot << " " << divresult.rem << "\n";
      itr = il + jump;
      if (x>p[*itr]) {
        if (dir==1) il = itr;
        if (dir==-1) ih = itr;
      } else {
        if (dir==1) ih = itr;
        if (dir==-1) il = itr;
      }
    } while ((ih-il)>1);

    return il;
  }
}

void insert(double *x, double **y, const int n, const int m,
            const double xi, const double *yi, const int n1, const int dir)
/* x is a 1D array of size n; \
 * y is a 2D array of size n x m;
 * yi is 1D array of size m;
 * n1 is the upperbound within x 
 * where xi to be inserted;
 * x is a 1D array of keys to array y;
 * x must be a sorted array;
  */
{
    if (n1 > n) return;
    if (xi > x[n1-1]) return;
    
  std::vector<int> myvector;
  std::vector<int>::iterator iam;
  double *tmpX,**tmpY;
  int tindex;
  int k = 0;

  tmpX = create(tmpX,n1,"insert: tmpX");
  tmpY = create(tmpY,n1,m,"insert: tmpY");
  
  if (xi < x[0]) {
      for (int i=0; i<n1; i++) {
          if (i == 0) {
              tmpX[i] = xi;
              for (int j=0; j<m; j++) tmpY[i][j] = yi[j];
          }
          else {
              tmpX[i] = x[k];
              for (int j=0; j<m; j++) tmpY[i][j] = y[k][j];
              k++;
          }
      }
      for (int i=0; i<n1; i++) {
        x[i] = tmpX[i];
        for (int j=0; j<m; j++) y[i][j] = tmpY[i][j];
      }
      return;
  }

  for (int i=0; i<n1; i++) myvector.push_back(i);

  iam = locate(xi,x,myvector.begin(),myvector.end(),dir);
  tindex = *iam;
  //std::cout << tindex << "\n";
  
  // Copy x and y to temporary variables in a given order;
  // Always do member by member copying for multidimensional arrays !!!; 
  for (int i=0; i<n1; i++) {
      if (i == tindex+1) {
          tmpX[i] = xi;
          for (int j=0; j<m; j++) tmpY[i][j] = yi[j];
      }
      else {
          tmpX[i] = x[k];
          for (int j=0; j<m; j++) tmpY[i][j] = y[k][j];
          k++;
      }
  }
  
  // Copy back the ordered data to x and y;
  // Always do member by member copying for multidimensional arrays !!!;
  for (int i=0; i<n1; i++) {
    x[i] = tmpX[i];
    for (int j=0; j<m; j++) y[i][j] = tmpY[i][j];
  }

  sfree(tmpX);
  destroy(tmpY);
  //for (iam=myvector.begin(); iam<myvector.end(); iam++) std::cout << *iam << " " << d[*iam] << "\n";
}
