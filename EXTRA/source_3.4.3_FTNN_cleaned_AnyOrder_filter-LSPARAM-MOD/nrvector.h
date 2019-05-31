#ifndef NRVECTOR_H
#define NRVECTOR_H

#include <stdio.h>

template <class T>
class NRvector;

template <class T>
std::ostream& operator<< ( std::ostream& os, const NRvector<T> &rhs );

template <class T>
class NRvector {
private:
  int nn;  // Size of array, indices 0..nn-1.
  T *v;    // Pointer to data array.
public:
  NRvector() {nn=0; v=NULL;}                      // Default constructor.
  explicit NRvector(int n);                       // Construct vector of size n.
  NRvector(int n, const T &a);                    // Initialize to constant value a.
  NRvector(int n, const T *a);                    // Initialize to values in C-style array a.
  NRvector(const NRvector &rhs);                  // Copy constructor.
  NRvector& operator=(const NRvector &rhs);       // Assignment operator.
  NRvector operator+(const NRvector &rhs);        // Addition operator.
  typedef T value_type;                           // Make T available.
  inline T& operator[](const int i);              // Return element number i.
  inline const T& operator[](const int i) const;  // const version.
  inline int size() const {return nn;}            // Return size of vector.
  void resize(int newn);                          // Resize, losing contents.
  void assign(int newn, const T &a);              // Resize and assign a to every element.
  void assign(int newn, const T *a);              // Resize and assign values of a.
  void assign(const T *a);                        // Assign values of a without checking bounds.
  void assign(const T a);                         // Assign a to every element.
  friend std::ostream& operator<< <>(std::ostream& os, const NRvector &rhs);
  ~NRvector();                                    // Destructor.
};

template <class T>
NRvector<T>::NRvector(int n) : nn(n), v(n>0 ? new T [n] : NULL) {}

template <class T>
NRvector<T>::NRvector(int n, const T &a)
{
  nn = n;
  if (nn > 0) {
    v = new T [nn];
    for (int i=0; i<nn; i++) v[i] = a;
  }
  else v = NULL;
}

template <class T>
NRvector<T>::NRvector(int n, const T *a)
{
  nn = n;
  if (nn > 0) {
    v = new T [nn];
    for (int i=0; i<nn; i++) v[i] = *(a++);
  } else v = NULL;
}

template <class T>
NRvector<T>::NRvector(const NRvector &rhs)
{
  nn = rhs.nn;
  if (nn > 0) {
    v = new T [nn];
    for (int i=0; i<nn; i++) v[i] = rhs.v[i];
  } else v = NULL;
}

template <class T>
NRvector<T>& NRvector<T>::operator=(const NRvector &rhs)
{
  if (this == &rhs) return *this;
  nn = rhs.nn;
  if (nn > 0) {
    v = new T [nn];
    for (int i=0; i<nn; i++) v[i] = rhs.v[i];
  } else v = NULL;
  return *this;
}

template <class T>
NRvector<T> NRvector<T>::operator+(const NRvector &rhs)
{
  if (nn == rhs.nn && nn > 0) {
    NRvector<T> tmp(rhs.nn);
    for (int i=0; i<nn; i++) tmp[i] = v[i] + rhs.v[i];
    return tmp;
  }
  return *this;
}

template <class T>
T& NRvector<T>::operator [] (const int i)
{
  if (i > nn-1 || i < 0) throw("index out of bound.");
  return v[i];
}

template <class T>
const T& NRvector<T>::operator[] (const int i) const
{
  if (i > nn-1 || i < 0) throw("index out of bound.");
  return v[i];
}

template <class T>
void NRvector<T>::resize(int newn)
{
  nn = newn;
  if (v) delete [] v;
  if (newn > 0)	v = new T [nn];
  else v = NULL;
}

template <class T>
void NRvector<T>::assign (int newn, const T &a)
{
    if (nn == newn && newn > 0) {
        for (int i=0; i<nn; i++) v[i] = a;
    } else if (newn) {
        nn = newn;
        if (v) delete [] v;
        v = new T [nn];
        for (int i=0; i<nn; i++) v[i] = a;
    } else v = NULL;
}

template <class T>
void NRvector<T>::assign (int newn, const T *a)
{
  if (nn == newn && newn > 0) {
    for (int i=0; i<nn; i++) v[i] = *(a++);
  } else if (newn > 0) {
    nn = newn;
    if (v) delete [] v;
    v = new T [nn];
    for (int i=0; i<nn; i++) v[i] = *(a++);
  } else v = NULL;
}

template <class T>
void NRvector<T>::assign (const T *a)
{
  if (nn > 0)	{
    for (int i=0; i<nn; i++) v[i] = a[i];
  }
  else v = NULL;
}

template <class T>
void NRvector<T>::assign(const T a)
{
  if (nn > 0) {
    for (int i=0; i<nn; i++) v[i] = a;
  }
}

// output operator
template <class T>
std::ostream& operator<< ( std::ostream& os, const NRvector<T> &rhs )
{
  os << "vector [" << rhs.nn << "] = \n";

  for ( int i = 0; i < rhs.nn; i++ ) os << " " << rhs[i];

  os << "\n";

  // must return the stream so we can chain output operations
  return os;
}

template <class T>
NRvector<T>::~NRvector ()
{
  delete [] v;
}

typedef NRvector<int> VecInt, VecInt_O, VecInt_IO;
typedef const NRvector<int> VecInt_I;
typedef NRvector<double> VecDoub, VecDoub_O, VecDoub_IO;
typedef const NRvector<double> VecDoub_I;

#endif // NRVECTOR_H

