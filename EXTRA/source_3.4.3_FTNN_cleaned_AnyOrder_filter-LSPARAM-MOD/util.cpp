#include <math.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "util.h"

void errmsg(const char *msg, const char *file, const char *func, const int line)
{
  fprintf(stderr,"ERROR: %s in file %s (function: %s) at line %d\n",msg,file,func,line);
}

bool search_file(const char *fname)
{
  bool found = false;
  DIR *dirFile = opendir("./");
  if (dirFile)
  {
    struct dirent* hFile;

    while (( hFile = readdir( dirFile )) != NULL )
    {
      if ( !strcmp( hFile->d_name, "."  )) continue;
      if ( !strcmp( hFile->d_name, ".." )) continue;

      // in linux hidden files all start with '.'
      if ( hFile->d_name[0] == '.' ) continue;

      if (strcmp(hFile->d_name,fname) == 0) {
	found = true;
	hFile = NULL;
      }
    }
  }
  closedir( dirFile );

  return found;
}

int compareAbsMaxArray(const int n, const double *x, const double y, double &xout)
{
  int count = 0;

  if (x == NULL) {
    perror("array empty");
    count = -1;
  } else {
    double tmp = y;
    for (int i=0; i<n; i++) {
      if (fabs(x[i]) > y) {
	count++;
	tmp = (fabs(x[i])>fabs(tmp))?x[i]:tmp;
      }
    }
    xout = tmp;
  }

  return count;
}


int compareAbsMaxVec(const VecDoub &x, const double y, double &xout)
{
  int count = 0;

  if (x.size() == 0) {
    perror("array empty");
    count = -1;
  } else {
    double tmp = y;
    for (int i=0; i<x.size(); i++) {
      if (fabs(x[i]) > y) {
	count++;
	tmp = (fabs(x[i])>fabs(tmp))?x[i]:tmp;
      }
    }
    xout = tmp;
  }

  return count;
}
