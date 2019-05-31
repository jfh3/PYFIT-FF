#ifndef _NRMATRIX_H_
#define _NRMATRIX_H_

#include <iostream>
#include <limits>
#include <stdio.h>

struct _error{
  const char *msg,
  *function,
  *file;
  unsigned long line;
};

//macro-like inline functions

/*template<class T>
inline T SQR(const T a) {return a*a;}

template<class T>
inline const T &MAX(const T &a, const T &b)
        {return b > a ? (b) : (a);}

inline float MAX(const double &a, const float &b)
        {return b > a ? (b) : float(a);}

inline float MAX(const float &a, const double &b)
        {return b > a ? float(b) : (a);}

template<class T>
inline const T &MIN(const T &a, const T &b)
        {return b < a ? (b) : (a);}

inline float MIN(const double &a, const float &b)
        {return b < a ? (b) : float(a);}

inline float MIN(const float &a, const double &b)
        {return b < a ? float(b) : (a);}

template<class T>
inline T SIGN(const T &a, const T &b)
	{return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}

inline float SIGN(const float &a, const double &b)
	{return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}

inline float SIGN(const double &a, const float &b)
	{return (float)(b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a));}

template<class T>
inline void SWAP(T &a, T &b)
{T dum=a; a=b; b=dum;}

// exception handling

#ifndef _USENRERRORCLASS_
#define throw(message) \
{printf("ERROR: %s\n     in file %s at line %d\n", message,__FILE__,__LINE__); throw(1);}
#else
struct NRerror {
	char *message;
	char *file;
	int line;
	NRerror(char *m, char *f, int l) : message(m), file(f), line(l) {}
};
#define throw(message) throw(NRerror(message,__FILE__,__LINE__));
void NRcatch(NRerror err) {
	printf("ERROR: %s\n     in file %s at line %d\n",
		err.message, err.file, err.line);
	exit(1);
}
#endif

// usage example:
//
//	try {
//		somebadroutine();
//	}
//	catch(NRerror s) {NRcatch(s);}
//
// (You can of course substitute any other catch body for NRcatch(s).)
*/

// Matrix Class

template <class T>
class NRmatrix;

template <class T>
std::ostream& operator<< ( std::ostream& os, const NRmatrix<T> &rhs );

template <class T>
class NRmatrix {
private:
  int nn;
  int mm;
  T **v;
public:
  NRmatrix();
  NRmatrix(int n, int m);			        // Zero-based array
  NRmatrix(int n, int m, const T &a);	                // Initialize to constant
  NRmatrix(int n, int m, const T *a);	                // Initialize to array by filling columns first.
  NRmatrix(const NRmatrix &rhs);		        // Copy constructor
  NRmatrix & operator=(const NRmatrix &rhs);	        // assignment
  typedef T value_type;                                 // make T available externally
  inline T* operator[](const int i);	                // subscripting: pointer to row i
  inline const T* operator[](const int i) const;
  inline int nrows() const;
  inline int ncols() const;
  void resize(int newn, int newm);                // resize (contents not preserved)
  void grow(int newn, int newm);                  // resize (contents within newn x newm preserved)
  void assign(int newn, int newm, const T &a);    // resize and assign a constant value
  inline void assign(const T &a);                 // Assign a constant value to all elements.
  inline void setrow(const int row, const T &a);  // Set a constant value to a specified row.
  inline void setcol(const int col, const T &a);  // Set a constant value to a specified column.
  // Modification operators
  NRmatrix& operator+= (const NRmatrix &rhs);
  NRmatrix& operator+=(const T d);
  NRmatrix operator+ (const NRmatrix &rhs);
  NRmatrix operator+ (const T d);
  NRmatrix& operator-= (const NRmatrix &rhs);
  NRmatrix& operator-= (const T d);
  NRmatrix operator- (const T d);
  NRmatrix operator- (const NRmatrix &rhs);
  NRmatrix& operator*= (const NRmatrix &rhs);
  NRmatrix& operator*= (const T a);
  NRmatrix operator* (const NRmatrix &rhs);
  NRmatrix operator* (const T a);
  //NRmatrix operator/ (double k) throw (message);
  // Computation operators
  NRmatrix<T> rowAvg();           // Compute average of each row.
  NRmatrix<T> colAvg();           // Compute average of each column.
  T det() const;                  // Compute determinant of 3x3 matrix.
  NRmatrix<T> inv() const;              // Compute inverse if it is nonsingular.
  void transpose();               // Resizes the matrix and overwrites the content.
  ~NRmatrix();
  friend std::ostream& operator<< <>(std::ostream& os, const NRmatrix &m);
};

template <class T>
NRmatrix<T>::NRmatrix() : nn(0), mm(0), v(NULL) {}

template <class T>
NRmatrix<T>::NRmatrix(int n, int m) : nn(n), mm(m), v(n>0 ? new T*[n] : NULL)
{
  int i,nel=m*n;
  if (v) v[0] = nel>0 ? new T[nel] : NULL;
  for (i=1;i<n;i++) v[i] = v[i-1] + m;
}

template <class T>
NRmatrix<T>::NRmatrix(int n, int m, const T &a) : nn(n), mm(m), v(n>0 ? new T*[n] : NULL)
{
  int i,j,nel=m*n;
  if (v) v[0] = nel>0 ? new T[nel] : NULL;
  for (i=1; i< n; i++) v[i] = v[i-1] + m;
  for (i=0; i< n; i++) for (j=0; j<m; j++) v[i][j] = a;
}

template <class T>
NRmatrix<T>::NRmatrix(int n, int m, const T *a) : nn(n), mm(m), v(n>0 ? new T*[n] : NULL)
{
  int i,j,nel=m*n;
  if (v) v[0] = nel>0 ? new T[nel] : NULL;
  for (i=1; i< n; i++) v[i] = v[i-1] + m;
  for (i=0; i< n; i++) for (j=0; j<m; j++) v[i][j] = *a++;
}

template <class T>
NRmatrix<T>::NRmatrix(const NRmatrix &rhs) : nn(rhs.nn), mm(rhs.mm), v(nn>0 ? new T*[nn] : NULL)
{
  int i,j,nel=mm*nn;
  if (v) v[0] = nel>0 ? new T[nel] : NULL;
  for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
  for (i=0; i< nn; i++) for (j=0; j<mm; j++) v[i][j] = rhs[i][j];
}

template <class T>
NRmatrix<T> & NRmatrix<T>::operator=(const NRmatrix<T> &rhs)
// postcondition: normal assignment via copying has been performed;
//		if matrix and rhs were different sizes, matrix
//		has been resized to match the size of rhs
{
  if (this != &rhs) {
    int i,j,nel;
    if (nn != rhs.nn || mm != rhs.mm) {
      if (v != NULL) {
	delete[] (v[0]);
	delete[] (v);
      }
      nn=rhs.nn;
      mm=rhs.mm;
      v = nn>0 ? new T*[nn] : NULL;
      nel = mm*nn;
      if (v) v[0] = nel>0 ? new T[nel] : NULL;
      for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
    }
    for (i=0; i< nn; i++) for (j=0; j<mm; j++) v[i][j] = rhs[i][j];
  }
  return *this;
}

template <class T>
inline T* NRmatrix<T>::operator[](const int i)	//subscripting: pointer to row i
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=nn) {
  throw("NRmatrix subscript out of bounds");
}
#endif
return v[i];
}

template <class T>
inline const T* NRmatrix<T>::operator[](const int i) const
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=nn) {
  throw("NRmatrix subscript out of bounds");
}
#endif
return v[i];
}

template <class T>
inline int NRmatrix<T>::nrows() const
{
  return nn;
}

template <class T>
inline int NRmatrix<T>::ncols() const
{
  return mm;
}

template <class T>
void NRmatrix<T>::resize(int newn, int newm)
{
  int i,nel;
  if (newn != nn || newm != mm) {
    if (v != NULL) {
      delete[] (v[0]);
      delete[] (v);
    }
    nn = newn;
    mm = newm;
    v = nn>0 ? new T*[nn] : NULL;
    nel = mm*nn;
    if (v) v[0] = nel>0 ? new T[nel] : NULL;
    for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
  }
}

template <class T>
void NRmatrix<T>::grow(int newn, int newm)
{
  int nr, nc;
  if (newn != nn || newm != mm) {
    NRmatrix<T> tmp = *this;
    assign(newn,newm,0.0);
    nr = (tmp.nn < newn ? tmp.nn : newn);
    nc = (tmp.mm < newm ? tmp.mm : newm);
    //std::cout << nr << " " << nc << "\n";
    //std::cout << *this;
    for (int i=0; i<nr; i++) for (int j=0; j<nc; j++) v[i][j] = tmp[i][j];
  }
}

template <class T>
void NRmatrix<T>::assign(int newn, int newm, const T& a)
{
  int i,j,nel;
  if (newn != nn || newm != mm) {
    if (v != NULL) {
      delete[] (v[0]);
      delete[] (v);
    }
    nn = newn;
    mm = newm;
    v = nn>0 ? new T*[nn] : NULL;
    nel = mm*nn;
    if (v) v[0] = nel>0 ? new T[nel] : NULL;
    for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
  }
  for (i=0; i< nn; i++) for (j=0; j<mm; j++) v[i][j] = a;
}

template <class T>
inline void NRmatrix<T>::assign(const T &a)
{
  if (nn > 0 && mm > 0) {
    int i, j;
    for (i=0; i< nn; i++) for (j=0; j<mm; j++) v[i][j] = a;
  }
}

template <class T>
inline void NRmatrix<T>::setrow(const int row, const T &a)
{
  int j;
  if (row >= 0 && row < nn) for (j=0; j<mm; j++) v[row][j] = a;
  else printf("Warning: row exceeds the bound .... [%s:%d]\n",
	      __FILE__, __LINE__);
}

template <class T>
inline void NRmatrix<T>::setcol(const int col, const T &a)
{
  int i;
  if (col >=0 && col < mm) for (i=0; i<nn; i++) v[i][col] = a;
  else printf("Warning: column exceeds the bound .... [%s:%d]\n",
	      __FILE__, __LINE__);
}

template <class T>
NRmatrix<T>& NRmatrix<T>::operator+= (const NRmatrix<T>& rhs)
{
  if (nn == rhs.nn && mm == rhs.mm) {
    for (int i=0; i<nn; i++) {
      for (int j=0; j<mm; j++) v[i][j] += rhs[i][j];
    }
  } else {
    printf("Error:matrices are not commensurate\n in file %s at line %d\n",__FILE__,__LINE__);
  }
  return *this;
}

template <class T>
NRmatrix<T>& NRmatrix<T>::operator+=(const T d)
{
  for (int i=0; i<nn; i++) for (int j=0; j<mm; j++) v[i][j] += d;
  return *this;
}

template <class T>
NRmatrix<T> NRmatrix<T>::operator+ (const NRmatrix<T> &rhs)
{
  NRmatrix<T> tmp(nn,mm);
  if (nn == rhs.nn && mm == rhs.mm) {
    for (int i=0; i<nn; i++) {
      for (int j=0; j<mm; j++) tmp[i][j] = v[i][j] + rhs[i][j];
    }
  } else {
    printf("Error:matrices are not commensurate\n in file %s at line %d\n",__FILE__,__LINE__);
  }
  return tmp;
}

template <class T>
NRmatrix<T> NRmatrix<T>::operator+ (const T d)
{
  NRmatrix<T> tmp(nn,mm);
  for (int i=0; i<nn; i++) {
    for (int j=0; j<mm; j++) tmp[i][j] = v[i][j] + d;
  }
  return tmp;
}

template <class T>
NRmatrix<T>& NRmatrix<T>::operator-= (const NRmatrix<T>& rhs)
{
  if (nn == rhs.nn && mm == rhs.mm) {
    for (int i=0; i<nn; i++) {
      for (int j=0; j<mm; j++) v[i][j] -= rhs[i][j];
    }
  } else {
    printf("Error:matrices are not commensurate\n in file %s at line %d\n",__FILE__,__LINE__);
  }
  return *this;
}

template <class T>
NRmatrix<T>& NRmatrix<T>::operator-=(const T d)
{
  for (int i=0; i<nn; i++) for (int j=0; j<mm; j++) v[i][j] -= d;
  return *this;
}

template <class T>
NRmatrix<T> NRmatrix<T>::operator- (const NRmatrix<T> &rhs)
{
  NRmatrix<T> tmp(nn,mm);
  if (nn == rhs.nn && mm == rhs.mm) {
    for (int i=0; i<nn; i++) {
      for (int j=0; j<mm; j++) tmp[i][j] = v[i][j] - rhs[i][j];
    }
    return tmp;
  } else {
    printf("Error:matrices are not commensurate\n in file %s at line %d\n",__FILE__,__LINE__);
  }
  return *this;
}

template <class T>
NRmatrix<T> NRmatrix<T>::operator- (const T d)
{
  NRmatrix<T> tmp(nn,mm);
  for (int i=0; i<nn; i++) {
    for (int j=0; j<mm; j++) tmp[i][j] = v[i][j] - d;
  }
  return tmp;
}

template <class T>
NRmatrix<T>& NRmatrix<T>::operator*= (const NRmatrix<T> &rhs)
{
  NRmatrix<T> tmp = *this;
  if (tmp.mm == rhs.nn) {
    T sum;
    resize(nn,rhs.mm);
    for (int i=0; i<nn; i++) {
      for (int j=0; j<rhs.mm; j++) {
	sum = 0.0;
	for (int k=0; k<mm; k++) sum += tmp[i][k] * rhs[k][j];
	v[i][j] = sum;
      }
    }
  } else {
    printf("Error: matrices are not compatible .... [%s:%d]\n", __FILE__, __LINE__);
  }
  return *this;
}

template <class T>
NRmatrix<T> NRmatrix<T>::operator* (const NRmatrix<T> &rhs)
{
  NRmatrix<T> tmp(nn,rhs.mm);
  if (mm == rhs.nn) {
    T sum;
    for (int i=0; i<nn; i++) {
      for (int j=0; j<rhs.mm; j++) {
	sum = 0.0;
	for (int k=0; k<mm; k++) sum += v[i][k] * rhs[k][j];
	tmp[i][j] = sum;
      }
    }
    return tmp;
  } else {
    printf("Error: matrices are not compatible .... [%s:%d]\n", __FILE__, __LINE__);
  }
  return *this;
}

template <class T>
NRmatrix<T> NRmatrix<T>::operator* (const T a)
{
  NRmatrix<T> tmp(nn,mm);
  for (int i=0; i<nn; i++) for (int j=0; j<mm; j++) tmp[i][j] = v[i][j] * a;
  return tmp;
}

template <class T>
NRmatrix<T>& NRmatrix<T>::operator*= (const T a)
{
  for (int i=0; i<nn; i++) for (int j=0; j<mm; j++) v[i][j] *= a;
  return *this;
}

template <class T>
NRmatrix<T> NRmatrix<T>::rowAvg()
{
  NRmatrix<T> tmp(nn,1,0.0);
  for (int i=0; i<nn; i++) {
    for (int j=0; j<mm; j++) tmp[i][0] += v[i][j];
    tmp[i][0] /= mm;
  }
  return tmp;
}

template <class T>
NRmatrix<T> NRmatrix<T>::colAvg()
{
  NRmatrix<T> tmp(1,mm,0.0);
  for (int j=0; j<mm; j++) {
    for (int i=0; i<nn; i++) tmp[0][j] += v[i][j];
    tmp[0][j] /= nn;
  }
  return tmp;
}

template <class T>
T NRmatrix<T>::det() const
{
  // Compute determinant of 2x2 or 3x3 matrices.

  // A =
  // | a11 a12 a13 |
  // | a21 a22 a23 |
  // | a31 a32 a33 |
  //
  // |A| = a11 * (a22 * a33 - a32 * a23)
  //       - a12 * (a21 * a33 - a31 * a23)
  //       + a13 * (a21 * a32 - a31 * a22)

  T tmp;

  if (nn == mm && nn == 2) {
    tmp = v[0][0] * v[1][1] - v[1][0] * v[0][1];
    return tmp;
  } else {
    if (nn == mm && nn == 3) {
      tmp = v[0][0] * (v[1][1] * v[2][2] - v[2][1] * v[1][2]);
      tmp += - v[0][1] * (v[1][0] * v[2][2] - v[2][0] * v[1][2]);
      tmp += v[0][2] * (v[1][0] * v[2][1] - v[2][0] * v[1][1]);
      return tmp;
    }
  }

  printf("Error: no method to compute determinant of %d x %d matrix .... [%s:%d]\n",
	 nn, mm, __FILE__, __LINE__);
  exit(0);

  return tmp;
}

template <class T>
NRmatrix<T> NRmatrix<T>::inv() const
{
  T det_val = this->det();
  NRmatrix<T> tmp(nn, mm);

  if (det_val != 0) {
    if (nn == 2) {
      // A =
      //    | a b |
      //    | c d |
      //
      tmp[0][0] = v[1][1] / det_val;
      tmp[0][1] = -v[0][1] / det_val;
      tmp[1][0] = -v[1][0] / det_val;
      tmp[1][1] = v[0][0] / det_val;
    }

    if (nn == 3) {
      // Row 1:
      // | a22 aq23 | | a13 a12 | | a12 a13 |
      // | a32 a33 | | a33 a32 | | a22 a23 |
      tmp[0][0] = (v[1][1] * v[2][2] - v[2][1] * v[1][2]) / det_val;
      tmp[0][1] = (v[0][2] * v[2][1] - v[2][2] * v[0][1]) / det_val;
      tmp[0][2] = (v[0][1] * v[1][2] - v[1][1] * v[0][2]) / det_val;
      // Row 2:
      // | a23 a21 | | a11 a13 | | a13 a11 |
      // | a33 a31 | | a31 a33 | | a23 a21 |
      tmp[1][0] = (v[1][2] * v[2][0] - v[2][2] * v[1][0]) / det_val;
      tmp[1][1] = (v[0][0] * v[2][2] - v[2][0] * v[0][2]) / det_val;
      tmp[1][2] = (v[0][2] * v[1][0] - v[1][2] * v[0][0]) / det_val;
      // Row 3:
      // | a21 a22 | | a12 a11 | | a11 a12 |
      // | a31 a32 | | a32 a31 | | a21 a22 |
      tmp[2][0] = (v[1][0] * v[2][1] - v[2][0] * v[1][1]) / det_val;
      tmp[2][1] = (v[0][1] * v[2][0] - v[2][1] * v[0][0]) / det_val;
      tmp[2][2] = (v[0][0] * v[1][1] - v[1][0] * v[0][1]) / det_val;
    }

    return tmp;
  } else {
    printf("Error: matrix is singular .... [%s:%d]\n", __FILE__, __LINE__);
    exit(0);
  }

  return tmp;
}


template <class T>
void NRmatrix<T>::transpose()
{
  NRmatrix<T> tmp = *this;
  if (v) {
    delete[] (v[0]);
    delete[] (v);
  }
  nn = tmp.mm;
  mm = tmp.nn;
  v = nn>0 ? new T*[nn] : NULL;
  int nel = nn*mm;
  if (v) v[0] = nel>0 ? new T[nel] : NULL;
  for (int i=1;i<nn;i++) v[i] = v[i-1] + mm;
  for (int i=0; i<nn; i++) for (int j=0; j<mm; j++) v[i][j] = tmp[j][i];
}

template <class T>
NRmatrix<T>::~NRmatrix()
{
  if (v != NULL) {
    delete[] (v[0]);
    delete[] (v);
  }
}

template <class T>
std::ostream& operator<< ( std::ostream& os, const NRmatrix<T> &m )
{
  os << "matrix [" << m.nrows() << "," << m.ncols() << "] = \n";
  if (m.nrows()*m.ncols() > 400) {
    os << " matrix is too large to show !!!\n";
  } else {
    for ( int i = 0; i < m.nrows(); i++ ) {
      for ( int j = 0; j < m.ncols(); j++ ) {
	os << "  " << m[i][j];
      }

      os << "\n";
    }
  }

  // must return the stream so we can chain output operations
  return os;
}


template <class T>
class NRMat3d {
private:
	int nn;
	int mm;
	int kk;
	T ***v;
public:
	NRMat3d();
	NRMat3d(int n, int m, int k);
	inline T** operator[](const int i);	//subscripting: pointer to row i
	inline const T* const * operator[](const int i) const;
	inline int dim1() const;
	inline int dim2() const;
	inline int dim3() const;
	~NRMat3d();
};

template <class T>
NRMat3d<T>::NRMat3d(): nn(0), mm(0), kk(0), v(NULL) {}

template <class T>
NRMat3d<T>::NRMat3d(int n, int m, int k) : nn(n), mm(m), kk(k), v(new T**[n])
{
  int i,j;
  v[0] = new T*[n*m];
  v[0][0] = new T[n*m*k];
  for(j=1; j<m; j++) v[0][j] = v[0][j-1] + k;
  for(i=1; i<n; i++) {
    v[i] = v[i-1] + m;
    v[i][0] = v[i-1][0] + m*k;
    for(j=1; j<m; j++) v[i][j] = v[i][j-1] + k;
  }
}

template <class T>
inline T** NRMat3d<T>::operator[](const int i) //subscripting: pointer to row i
{
  return v[i];
}

template <class T>
inline const T* const * NRMat3d<T>::operator[](const int i) const
{
  return v[i];
}

template <class T>
inline int NRMat3d<T>::dim1() const
{
  return nn;
}

template <class T>
inline int NRMat3d<T>::dim2() const
{
  return mm;
}

template <class T>
inline int NRMat3d<T>::dim3() const
{
  return kk;
}

template <class T>
NRMat3d<T>::~NRMat3d()
{
  if (v != NULL) {
    delete[] (v[0][0]);
    delete[] (v[0]);
    delete[] (v);
  }
}

// basic type names (redefine if your bit lengths don't match)

typedef int Int; // 32 bit integer
typedef unsigned int Uint;

#ifdef _MSC_VER
typedef __int64 Llong; // 64 bit integer
typedef unsigned __int64 Ullong;
#else
typedef long long int Llong; // 64 bit integer
typedef unsigned long long int Ullong;
#endif

typedef char Char; // 8 bit integer
typedef unsigned char Uchar;

typedef double Doub; // default floating type
typedef long double Ldoub;

//typedef complex<double> Complex; // default complex type

typedef bool Bool;

// NaN: uncomment one of the following 3 methods of defining a global NaN
// you can test by verifying that (NaN != NaN) is true

static const Doub NaN = std::numeric_limits<Doub>::quiet_NaN();

//Uint proto_nan[2]={0xffffffff, 0x7fffffff};
//double NaN = *( double* )proto_nan;

//Doub NaN = sqrt(-1.);

// matrix types

typedef const NRmatrix<Int> MatInt_I;
typedef NRmatrix<Int> MatInt, MatInt_O, MatInt_IO;

typedef const NRmatrix<Uint> MatUint_I;
typedef NRmatrix<Uint> MatUint, MatUint_O, MatUint_IO;

typedef const NRmatrix<Llong> MatLlong_I;
typedef NRmatrix<Llong> MatLlong, MatLlong_O, MatLlong_IO;

typedef const NRmatrix<Ullong> MatUllong_I;
typedef NRmatrix<Ullong> MatUllong, MatUllong_O, MatUllong_IO;

typedef const NRmatrix<Char> MatChar_I;
typedef NRmatrix<Char> MatChar, MatChar_O, MatChar_IO;

typedef const NRmatrix<Uchar> MatUchar_I;
typedef NRmatrix<Uchar> MatUchar, MatUchar_O, MatUchar_IO;

typedef const NRmatrix<Doub> MatDoub_I;
typedef NRmatrix<Doub> MatDoub, MatDoub_O, MatDoub_IO;

typedef const NRmatrix<Bool> MatBool_I;
typedef NRmatrix<Bool> MatBool, MatBool_O, MatBool_IO;

// 3D matrix types

typedef const NRMat3d<Doub> Mat3DDoub_I;
typedef NRMat3d<Doub> Mat3DDoub, Mat3DDoub_O, Mat3DDoub_IO;

// Floating Point Exceptions for Microsoft compilers

#ifdef _TURNONFPES_
#ifdef _MSC_VER
struct turn_on_floating_exceptions {
  turn_on_floating_exceptions() {
    int cw = _controlfp( 0, 0 );
    cw &=~(EM_INVALID | EM_OVERFLOW | EM_ZERODIVIDE );
    _controlfp( cw, MCW_EM );
  }
};
turn_on_floating_exceptions yes_turn_on_floating_exceptions;
#endif /* _MSC_VER */
#endif /* _TURNONFPES */

#endif /* _NRMATRIX_H_ */

