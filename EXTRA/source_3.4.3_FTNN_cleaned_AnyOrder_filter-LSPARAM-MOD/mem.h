#ifndef MEM_H
#define MEM_H

void *smalloc(int nbytes, const char *name);
void *srealloc(void *ptr, int nbytes, const char *name);
void sfree(void *ptr);

/* -----------------------------------------
   create 1d array
   ----------------------------------------- */

template <typename T>
T *create(T *&array, int n, const char *name)
{
  int nbytes = ((int) sizeof(T)) * n;
  array = (T *) smalloc(nbytes,name);
  return array;
}

/* ----------------------------------------------------------------------
   grow or shrink 1d array
   ---------------------------------------------------------------------- */

template <typename T>
T *grow(T *&array, int n, const char *name)
{
  if (array == NULL) return create(array,n,name);

  int nbytes = ((int) sizeof(T)) * n;
  array = (T *) srealloc(array,nbytes,name);
  return array;
}

/* ----------------------------------------------------------------------
   create a 2d array
------------------------------------------------------------------------- */

template <typename TYPE>
TYPE **create(TYPE **&array, int n1, int n2, const char *name)
{
    int nbytes = ((int) sizeof(TYPE)) * n1 * n2;
    TYPE *data = (TYPE *) smalloc(nbytes,name);
    nbytes = ((int) sizeof(TYPE *)) * n1;
    array = (TYPE **) smalloc(nbytes,name);

    int n = 0;
    for (int i = 0; i < n1; i++) {
        array[i] = &data[n];
        n += n2;
    }
    return array;
}

/* ----------------------------------------------------------------------
   grow or shrink 1st dim of a 2d array
   last dim must stay the same
------------------------------------------------------------------------- */

template <typename TYPE>
TYPE **grow(TYPE **&array, int n1, int n2, const char *name)
{
    if (array == NULL) return create(array,n1,n2,name);

    int nbytes = ((int) sizeof(TYPE)) * n1*n2;
    TYPE *data = (TYPE *) srealloc(array[0],nbytes,name);
    nbytes = ((int) sizeof(TYPE *)) * n1;
    array = (TYPE **) srealloc(array,nbytes,name);

    int n = 0;
    for (int i = 0; i < n1; i++) {
        array[i] = &data[n];
        n += n2;
    }
    return array;
}

/* ----------------------------------------------------------------------
   destroy a 2d array
------------------------------------------------------------------------- */

template <typename TYPE>
void destroy(TYPE **array)
{
    if (array == NULL) return;
    sfree(array[0]);
    sfree(array);
}

/* ----------------------------------------------------------------------
   create a 3d array
------------------------------------------------------------------------- */

template <typename TYPE>
TYPE ***create(TYPE ***&array, int n1, int n2, int n3, const char *name)
{
  array = (TYPE ***) smalloc(n1*sizeof(TYPE *),name);
  for (int i=0; i<n1; i++) array[i] = (TYPE **) smalloc(n2*sizeof(TYPE *),name);
  for (int i=0; i<n1; i++) {
    for (int j=0; j<n2; j++) array[i][j] = (TYPE *) smalloc(n3*sizeof(TYPE),name);
  }
  return array;
}

template <typename TYPE>
TYPE ***create3d(TYPE ***&array, int n1, int n2, int n3, const char *name)
{
  int nbytes = ((int) sizeof(TYPE)) *n1*n2*n3;
  TYPE *data = (TYPE *) smalloc(nbytes,name);
  nbytes = ((int) sizeof(TYPE *)) * n1 * n2;
  TYPE **ptr = (TYPE **) smalloc(nbytes,name);
  array = (TYPE ***) smalloc(n1*sizeof(TYPE *),name);
  int n = 0;
  for (int i=0; i<n1; i++) {
    array[i] = &ptr[n];
    n += n2;
  }
  n = 0;
  for (int i=0; i<n1; i++) {
    for (int j=0; j<n2; j++) {
      array[i][j] = &data[n];
      n += n3;
    }
  }
  return array;
}

/* ----------------------------
 * destroy 3d array
 * ---------------------------- */

template < typename TYPE>
void destroy(TYPE ***array, int n1, int n2, int n3)
{
  if (array == NULL) return;
  for (int i=0; i<n1; i++) {
    for (int j=0; j<n2; j++) free(array[i][j]);
    free(array[i]);
  }
  free(array);
}

template < typename TYPE>
void destroy3d(TYPE ***array)
{
  if (array == NULL) return;
  free(array[0][0]);
  free(array[0]);
  free(array);
}

#endif // MEM_H
