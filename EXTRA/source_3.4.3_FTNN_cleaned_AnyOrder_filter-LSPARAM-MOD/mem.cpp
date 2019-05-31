#include <cstdlib>
#include <cstdio>
#include "mem.h"

/* ----------------------------------------------------------------------
   safe malloc
------------------------------------------------------------------------- */

void *smalloc(int nbytes, const char *name)
{
  if (nbytes == 0) return NULL;
  void *ptr = malloc(nbytes);

  if (ptr == NULL) {
    char str[128];
    sprintf(str,"Failed to allocate %d for array %s",
            nbytes,name);
    fprintf(stderr,"%s\n",str);
  }
  return ptr;
}

/* ----------------------------------------------------------------------
   safe realloc
------------------------------------------------------------------------- */

void *srealloc(void *ptr, int nbytes, const char *name)
{
  if (nbytes == 0) {
    free(ptr);
    return NULL;
  }

  ptr = realloc(ptr,nbytes);
  if (ptr == NULL) {
    char str[128];
    sprintf(str,"Failed to reallocate %d bytes for array %s",
            nbytes,name);
    fprintf(stderr,"%s\n",str);
  }
  return ptr;
}

/* ----------------------------------------------------------------------
   safe free
------------------------------------------------------------------------- */

void sfree(void *ptr)
{
  if (ptr == NULL) return;
  free(ptr);
}


