#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "globals.h"
#include "crystal_struc.h"
#include "util.h"
#include "write.h"

void set_struc(const char *struc, const double scale, int &nb)
// It updates lattice and basis vectors of a selected structure
// into global variables trans0 and basis0 respectively and
// returns number of bases of the structure in nb.
{
  int ifound = 0;

  if (strcmp(struc,"dimer") == 0) {
    ifound = 1;
    dimer(scale,nb);
  }

  if (strcmp(struc,"fcc") == 0) {
    ifound = 1;
    fcc(scale,nb);
  }

  if (strcmp(struc,"bcc") == 0) {
    ifound = 1;
    bcc(scale,nb);
  }

  if (strcmp(struc,"A15") == 0) {
    ifound = 1;
    A15(scale,nb);
  }

  if (strcmp(struc,"diam") == 0) {
    ifound = 1;
    diam(scale,nb);
  }

  if (strcmp(struc,"A5") == 0) {
    ifound = 1;
    betatin(scale,nb);
  }

  if (strcmp(struc,"sc") == 0) {
    ifound = 1;
    sc(scale,nb);
  }

  if (strcmp(struc,"wurtzite") == 0) {
    ifound = 1;
    wurtzite(scale,nb);
  }

  if (strcmp(struc,"hex") == 0) {
    ifound = 1;
    hex(scale, nb);
  }

  if (strcmp(struc,"hcp") == 0) {
    ifound = 1;
    hcp(scale,nb);
  }

  if (strcmp(struc,"trimerD3h") == 0) {
    ifound = 1;
    trimerD3h(scale,nb);
  }

  if (strcmp(struc,"trimerDih") == 0) {
    ifound = 1;
    trimerD3h(scale,nb);
  }

  if (strcmp(struc,"trimerC2v") == 0) {
    ifound = 1;
    trimerC2v(scale,nb);
  }

  if (strcmp(struc,"tetramerDih") == 0) {
    ifound = 1;
    tetramerDih(scale,nb); // four-atom linear chain.
  }

  if (strcmp(struc,"tetramerD4h") == 0) {
    ifound = 1;
    tetramerD4h(scale,nb); // Square.
  }

  if (strcmp(struc,"tetramerTd") == 0) {
    ifound = 1;
    tetramerTd(scale,nb); // Tetrahedral.
  }

  if (strcmp(struc,"pentamerD5h") == 0) {
    ifound = 1;
    pentamerD5h(scale,nb);
  }

  if (strcmp(struc,"BC8") == 0) {
    ifound = 1;
    BC8(scale,nb);
  }

  if (strcmp(struc,"ST12") == 0) {
    ifound = 1;
    ST12(scale,nb);
  }

  if (strcmp(struc,"cP46") == 0) {
    ifound = 1;
    cP46(scale,nb);
  }

  if (strcmp(struc,"graphitic") == 0) {
    ifound = 1;
    graphitic(scale,nb);
  }

  if (!ifound) {
    char buf[256];
    sprintf(buf," cannot find structure: %s",struc);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }
}

void graphitic(const double scale, int &nb)
{
  double a = lc_a0*scale;
  double bb = 0.5*sqrt(3.0);

  nb = 2;

  if (basis0 == NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.5*a;
  trans0[1][1] = bb*a;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = 10.0*a;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = 0.5*a;
  basis0[1][1] = bb*a/3.0;
  basis0[1][2] = 0.0;
}

void hcp(const double scale, int &nb)
{
  double a, a2, bb, c;

  a = lc_a0*scale;
  a2 = a/2.0;
  bb = a2*sqrt(3.0);
  if (lc_c0 < 1.e-3) c = a*sqrt(8.0/3.0);
  else c = lc_c0*scale;
  nb = 2;

  if (basis0 == NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = a2;
  trans0[1][1] = bb;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = c;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = a2;
  basis0[1][1] = bb/3.0;
  basis0[1][2] = c/2.0;
}

void pentamerD5h (const double scale, int &nb)
{
  double a;
  double com_rc;

  com_rc = 1.5*Rc;
  if (PotentialType == 1) com_rc = Rc;

  a = lc_a0*scale;
  nb = 5;

  if (basis0 == NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = 5.0*com_rc;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = 5.0*com_rc;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = 5.0*com_rc;

  basis0[0][0] = -0.5*a;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = 0.5*a;
  basis0[1][1] = 0.0;
  basis0[1][2] = 0.0;

  basis0[2][0] = -0.5*a - a*cos(PI*72.0/180.0);
  basis0[2][1] = a*sin(PI*72.0/180.0);
  basis0[2][2] = 0.0;

  basis0[3][0] = 0.5*a + a*cos(PI*72.0/180.0);
  basis0[3][1] = a*sin(PI*72.0/180.0);
  basis0[3][2] = 0.0;

  basis0[4][0] = 0.0;
  basis0[4][1] = a*sin(PI*72.0/180.0) + a*cos(PI*54.0/180.0);
  basis0[4][2] = 0.0;
}

void tetramerTd (const double scale, int &nb)
{
  double a;
  double com_rc;

  com_rc = 1.5*Rc;
  if (PotentialType == 1) com_rc = Rc;

  a = lc_a0*sqrt(2.0)*scale;
  nb = 4;

  if (basis0 == NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = 5.0*com_rc;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = 5.0*com_rc;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = 5.0*com_rc;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = a*0.5;
  basis0[1][1] = a*0.5;
  basis0[1][2] = 0.0;

  basis0[2][0] = a*0.5;
  basis0[2][1] = 0.0;
  basis0[2][2] = a*0.5;

  basis0[3][0] = 0.0;
  basis0[3][1] = a*0.5;
  basis0[3][2] = a*0.5;
}

void tetramerDih (const double scale, int &nb) // four-atom linear chain.
{
  double a,com_rc;

  com_rc = 1.5*Rc;
  if (PotentialType == 1) com_rc = Rc;
  a = lc_a0*scale;
  nb = 4;

  if (basis0 == NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = 5.0*com_rc;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = 5.0*com_rc;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = 5.0*com_rc;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = a;
  basis0[1][1] = 0.0;
  basis0[1][2] = 0.0;

  basis0[2][0] = a + a;
  basis0[2][1] = 0.0;
  basis0[2][2] = 0.0;

  basis0[3][0] = a + a + a;
  basis0[3][1] = 0.0;
  basis0[3][2] = 0.0;
}

void tetramerD4h (const double scale, int &nb) // square.
{
  double a;
  double com_rc;

  com_rc = 1.5*Rc;
  if (PotentialType == 1) com_rc = Rc;

  a = lc_a0*scale;
  nb = 4;

  if (basis0 == NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = 5.0*com_rc;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = 5.0*com_rc;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = 5.0*com_rc;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = a;
  basis0[1][1] = 0.0;
  basis0[1][2] = 0.0;

  basis0[2][0] = 0.0;
  basis0[2][1] = a;
  basis0[2][2] = 0.0;

  basis0[3][0] = a;
  basis0[3][1] = a;
  basis0[3][2] = 0.0;
}

void trimerC2v (const double scale, int &nb)
{
  double a,theta,com_rc;

  com_rc = 1.5*Rc;
  if (PotentialType == 1) com_rc = Rc;

  theta = 77.8; // For Si3 in degrees
  a = lc_a0*scale;
  nb = 3;

  if (basis0 == NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = 5.0*com_rc;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = 5.0*com_rc;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = 5.0*com_rc;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = a*sin(0.5*theta*PI/180);
  basis0[1][1] = a*cos(0.5*theta*PI/180);
  basis0[1][2] = 0.0;

  basis0[2][0] = 2.0*a*sin(0.5*theta*PI/180);
  basis0[2][1] = 0.0;
  basis0[2][2] = 0.0;
}

void trimerD3h(const double scale, int &nb) // triangle
{
  double a, rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;
  a = lc_a0*scale;
  nb = 3;

  if (basis0 == NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = 5.0*rc;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = 5.0*rc;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = 5.0*rc;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = a;
  basis0[1][1] = 0.0;
  basis0[1][2] = 0.0;

  basis0[2][0] = a*0.5;
  basis0[2][1] = a*sqrt(3.0)*0.5;
  basis0[2][2] = 0.0;
}

void trimerDih(const double scale, int &nb) // chain
{
  double a, rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;
  a = lc_a0*scale;
  nb = 3;

  if (basis0 == NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = 5.0*rc;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = 5.0*rc;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = 5.0*rc;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = a;
  basis0[1][1] = 0.0;
  basis0[1][2] = 0.0;

  basis0[2][0] = a + a;
  basis0[2][1] = 0.0;
  basis0[2][2] = 0.0;
}

void hex(const double scale, int &nb)
{
  double a, a2, bb;

  a = lc_a0*scale;
  a2 = a/2.0;
  bb = a*sqrt(3.0)/2.0;
  nb = 1;

  if (basis0==NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = a2;
  trans0[1][1] = bb;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = a;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;
}

void A15(const double scale, int &nb)
{
  double a, a2, a4, a34;

  a = lc_a0*scale;
  a2 = a/2.0;
  a4 = a2/2.0;
  a34 = a2/2.0*3;
  nb = 8;

  if (basis0==NULL) basis0 = create(basis0,nb,3,"SetStruct:create()");
  else basis0 = grow(basis0,nb,3,"SetStruct:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = a;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = a;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = a2;
  basis0[1][1] = a2;
  basis0[1][2] = a2;

  basis0[2][0] = a4;
  basis0[2][1] = a2;
  basis0[2][2] = 0.0;

  basis0[3][0] = a34;
  basis0[3][1] = a2;
  basis0[3][2] = 0.0;

  basis0[4][0] = 0.0;
  basis0[4][1] = a4;
  basis0[4][2] = a2;

  basis0[5][0] = 0;
  basis0[5][1] = a34;
  basis0[5][2] = a2;

  basis0[6][0] = a2;
  basis0[6][1] = 0;
  basis0[6][2] = a4;

  basis0[7][0] = a2;
  basis0[7][1] = 0.0;
  basis0[7][2] = a34;
}

void fcc(const double scale, int &nb)
{
  double a;

  nb = 4;
  a = lc_a0*scale;

  if (!basis0) basis0 = create(basis0,nb,3,"fcc:create()");
  else basis0 = grow(basis0,nb,3,"fcc:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = a;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = a;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = 0.5*a;
  basis0[1][1] = 0.5*a;
  basis0[1][2] = 0.0;

  basis0[2][0] = 0.5*a;
  basis0[2][1] = 0.0;
  basis0[2][2] = 0.5*a;

  basis0[3][0] = 0.0;
  basis0[3][1] = 0.5*a;
  basis0[3][2] = 0.5*a;
}

void diam(const double scale, int &nb)
{
  double a;

  nb = 8;
  a = lc_a0*scale;

  if (!basis0) basis0 = create(basis0,nb,3,"diam:create()");
  else basis0 = grow(basis0,nb,3,"diam:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = a;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = a;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = 0.5*a;
  basis0[1][1] = 0.5*a;
  basis0[1][2] = 0.0;

  basis0[2][0] = 0.5*a;
  basis0[2][1] = 0.0;
  basis0[2][2] = 0.5*a;

  basis0[3][0] = 0.0;
  basis0[3][1] = 0.5*a;
  basis0[3][2] = 0.5*a;

  basis0[4][0] = 0.25*a;
  basis0[4][1] = 0.25*a;
  basis0[4][2] = 0.25*a;

  basis0[5][0] = 0.75*a;
  basis0[5][1] = 0.75*a;
  basis0[5][2] = 0.25*a;

  basis0[6][0] = 0.75*a;
  basis0[6][1] = 0.25*a;
  basis0[6][2] = 0.75*a;

  basis0[7][0] = 0.25*a;
  basis0[7][1] = 0.75*a;
  basis0[7][2] = 0.75*a;
}

void wurtzite(const double scale, int &nb)
{
  //double a, c;
  nb = 4;

  //a = lc_a0*scale;
  //c = lc_c0*scale;

  if (!basis0) basis0 = create(basis0,nb,3,"wurtzite:create()");
  else basis0 = grow(basis0,nb,3,"wurtzite:grow()");

  trans0[0][0] = 3.846203*scale; //a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = -1.92310100e+00*scale; //-a*0.5;
  trans0[1][1] = 3.33090900e+00*scale; //a*sqrt(3.0)*0.5;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = 6.36663900e+00*scale; //c;

  basis0[0][0] = -1.58976798e-06*scale; //0.0;
  basis0[0][1] = 2.22060711e+00*scale; //0.0;
  basis0[0][2] = 4.02017205e-01*scale; //0.0;

  basis0[1][0] = 1.92310359e+00*scale; //0.0;
  basis0[1][1] = 1.11030189e+00*scale; //0.0;
  basis0[1][2] = 3.58533670e+00*scale; //3.0*c/8.0;

  basis0[2][0] = -1.58976798e-06*scale; //0.5*a;
  basis0[2][1] = 2.22060711e+00*scale; //-0.5*a/sqrt(3.0);
  basis0[2][2] = 2.78130230e+00*scale; //0.5*c;

  basis0[3][0] = 1.92310359e+00*scale; //0.5*a;
  basis0[3][1] = 1.11030189e+00*scale; //-0.5*a/sqrt(3.0);
  basis0[3][2] = 5.96462180e+00*scale; //7.0*c/8.0;
}

void bcc(const double scale, int &nb)
{
  double a = lc_a0*scale;
  nb = 2;

  if (!basis0) basis0 = create(basis0,nb,3,"bcc:create()");
  else basis0 = grow(basis0,nb,3,"bcc:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = a;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = a;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = 0.5*a;
  basis0[1][1] = 0.5*a;
  basis0[1][2] = 0.5*a;
}

void sc(const double scale, int &nb)
{
  double a = lc_a0*scale;
  nb = 1;

  if (!basis0) basis0 = create(basis0,nb,3,"sc:create()");
  else basis0 = grow(basis0,nb,3,"sc:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = a;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = a;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;
}

void dimer(const double scale, int &nb)
{
  double a = lc_a0*scale;
  double rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;
  nb = 2;

  if (!basis0) basis0 = create(basis0,nb,3,"sc:create()");
  else basis0 = grow(basis0,nb,3,"sc:grow()");

  trans0[0][0] = 5.0*rc;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = 5.0*rc;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = 5.0*rc;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = a;
  basis0[1][1] = 0.0;
  basis0[1][2] = 0.0;
}

void betatin(const double scale, int &nb)
{
  double a = lc_a0*scale;
  double c = lc_c0*scale;
  double ca = c/a;

  nb = 4;

  if (!basis0) basis0 = create(basis0,nb,3,"betatin:create()");
  else basis0 = grow(basis0,nb,3,"betating:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = a;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = a*ca;

  basis0[0][0] = 0.0;
  basis0[0][1] = 0.0;
  basis0[0][2] = 0.0;

  basis0[1][0] = 0.0;
  basis0[1][1] = 0.5*a;
  basis0[1][2] = a*ca/4.0;

  basis0[2][0] = 0.5*a;
  basis0[2][1] = 0.5*a;
  basis0[2][2] = a*ca/2.0;

  basis0[3][0] = 0.5*a;
  basis0[3][1] = 0.0;
  basis0[3][2] = a*ca*3.0/4.0;
}

void cP46(const double scale, int &nb)
{
  double a = lc_a0*scale; // lattice constant and basis from www.materialsproject.org

  nb = 46;

  if (!basis0) basis0 = create(basis0,nb,3,"cP46:create()");
  else basis0 = grow(basis0,nb,3,"cP46:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = a;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = a;

  basis0[0][0]=0.000000*a;
  basis0[0][1]=0.882820*a;
  basis0[0][2]=0.307890*a;
  basis0[1][0]=0.000000*a;
  basis0[1][1]=0.117180*a;
  basis0[1][2]=0.692110*a;
  basis0[2][0]=0.000000*a;
  basis0[2][1]=0.117180*a;
  basis0[2][2]=0.307890*a;
  basis0[3][0]=0.000000*a;
  basis0[3][1]=0.882820*a;
  basis0[3][2]=0.692110*a;
  basis0[4][0]=0.307890*a;
  basis0[4][1]=0.000000*a;
  basis0[4][2]=0.882820*a;
  basis0[5][0]=0.382820*a;
  basis0[5][1]=0.500000*a;
  basis0[5][2]=0.192110*a;
  basis0[6][0]=0.692110*a;
  basis0[6][1]=0.000000*a;
  basis0[6][2]=0.117180*a;
  basis0[7][0]=0.617180*a;
  basis0[7][1]=0.500000*a;
  basis0[7][2]=0.807890*a;
  basis0[8][0]=0.307890*a;
  basis0[8][1]=0.000000*a;
  basis0[8][2]=0.117180*a;
  basis0[9][0]=0.617180*a;
  basis0[9][1]=0.500000*a;
  basis0[9][2]=0.192110*a;
  basis0[10][0]=0.692110*a;
  basis0[10][1]=0.000000*a;
  basis0[10][2]=0.882820*a;
  basis0[11][0]=0.382820*a;
  basis0[11][1]=0.500000*a;
  basis0[11][2]=0.807890*a;
  basis0[12][0]=0.882820*a;
  basis0[12][1]=0.307890*a;
  basis0[12][2]=0.000000*a;
  basis0[13][0]=0.500000*a;
  basis0[13][1]=0.807890*a;
  basis0[13][2]=0.617180*a;
  basis0[14][0]=0.117180*a;
  basis0[14][1]=0.692110*a;
  basis0[14][2]=0.000000*a;
  basis0[15][0]=0.500000*a;
  basis0[15][1]=0.192110*a;
  basis0[15][2]=0.382820*a;
  basis0[16][0]=0.882820*a;
  basis0[16][1]=0.692110*a;
  basis0[16][2]=0.000000*a;
  basis0[17][0]=0.500000*a;
  basis0[17][1]=0.192110*a;
  basis0[17][2]=0.617180*a;
  basis0[18][0]=0.117180*a;
  basis0[18][1]=0.307890*a;
  basis0[18][2]=0.000000*a;
  basis0[19][0]=0.500000*a;
  basis0[19][1]=0.807890*a;
  basis0[19][2]=0.382820*a;
  basis0[20][0]=0.192110*a;
  basis0[20][1]=0.382820*a;
  basis0[20][2]=0.500000*a;
  basis0[21][0]=0.807890*a;
  basis0[21][1]=0.617180*a;
  basis0[21][2]=0.500000*a;
  basis0[22][0]=0.192110*a;
  basis0[22][1]=0.617180*a;
  basis0[22][2]=0.500000*a;
  basis0[23][0]=0.807890*a;
  basis0[23][1]=0.382820*a;
  basis0[23][2]=0.500000*a;
  basis0[24][0]=0.250000*a;
  basis0[24][1]=0.500000*a;
  basis0[24][2]=0.000000*a;
  basis0[25][0]=0.750000*a;
  basis0[25][1]=0.500000*a;
  basis0[25][2]=0.000000*a;
  basis0[26][0]=0.000000*a;
  basis0[26][1]=0.250000*a;
  basis0[26][2]=0.500000*a;
  basis0[27][0]=0.000000*a;
  basis0[27][1]=0.750000*a;
  basis0[27][2]=0.500000*a;
  basis0[28][0]=0.500000*a;
  basis0[28][1]=0.000000*a;
  basis0[28][2]=0.250000*a;
  basis0[29][0]=0.500000*a;
  basis0[29][1]=0.000000*a;
  basis0[29][2]=0.750000*a;
  basis0[30][0]=0.816380*a;
  basis0[30][1]=0.816380*a;
  basis0[30][2]=0.816380*a;
  basis0[31][0]=0.183620*a;
  basis0[31][1]=0.183620*a;
  basis0[31][2]=0.183620*a;
  basis0[32][0]=0.183620*a;
  basis0[32][1]=0.183620*a;
  basis0[32][2]=0.816380*a;
  basis0[33][0]=0.183620*a;
  basis0[33][1]=0.816380*a;
  basis0[33][2]=0.183620*a;
  basis0[34][0]=0.316380*a;
  basis0[34][1]=0.316380*a;
  basis0[34][2]=0.683620*a;
  basis0[35][0]=0.816380*a;
  basis0[35][1]=0.816380*a;
  basis0[35][2]=0.183620*a;
  basis0[36][0]=0.816380*a;
  basis0[36][1]=0.183620*a;
  basis0[36][2]=0.816380*a;
  basis0[37][0]=0.683620*a;
  basis0[37][1]=0.683620*a;
  basis0[37][2]=0.316380*a;
  basis0[38][0]=0.816380*a;
  basis0[38][1]=0.183620*a;
  basis0[38][2]=0.183620*a;
  basis0[39][0]=0.683620*a;
  basis0[39][1]=0.683620*a;
  basis0[39][2]=0.683620*a;
  basis0[40][0]=0.183620*a;
  basis0[40][1]=0.816380*a;
  basis0[40][2]=0.816380*a;
  basis0[41][0]=0.316380*a;
  basis0[41][1]=0.316380*a;
  basis0[41][2]=0.316380*a;
  basis0[42][0]=0.316380*a;
  basis0[42][1]=0.683620*a;
  basis0[42][2]=0.316380*a;
  basis0[43][0]=0.683620*a;
  basis0[43][1]=0.316380*a;
  basis0[43][2]=0.683620*a;
  basis0[44][0]=0.683620*a;
  basis0[44][1]=0.316380*a;
  basis0[44][2]=0.316380*a;
  basis0[45][0]=0.316380*a;
  basis0[45][1]=0.683620*a;
  basis0[45][2]=0.683620*a;
}

void BC8(const double scale, int &nb)
{
  double a = lc_a0*scale; // lattice constant and bases from www.materialsproject.org

  nb = 16;

  if (!basis0) basis0 = create(basis0,nb,3,"BC8:create()");
  else basis0 = grow(basis0,nb,3,"BC8:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = a;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = a;

  basis0[0][0] = 0.398443*a;
  basis0[0][1] = 0.101557*a;
  basis0[0][2] = 0.898443*a;

  basis0[1][0] = 0.101557*a;
  basis0[1][1] = 0.101557*a;
  basis0[1][2] = 0.101557*a;

  basis0[2][0] = 0.101557*a;
  basis0[2][1] = 0.898443*a;
  basis0[2][2] = 0.398443*a;

  basis0[3][0] = 0.398443*a;
  basis0[3][1] = 0.898443*a;
  basis0[3][2] = 0.601557*a;

  basis0[4][0] = 0.601557*a;
  basis0[4][1] = 0.898443*a;
  basis0[4][2] = 0.101557*a;

  basis0[5][0] = 0.898443*a;
  basis0[5][1] = 0.898443*a;
  basis0[5][2] = 0.898443*a;

  basis0[6][0] = 0.398443*a;
  basis0[6][1] = 0.601557*a;
  basis0[6][2] = 0.101557*a;

  basis0[7][0] = 0.601557*a;
  basis0[7][1] = 0.101557*a;
  basis0[7][2] = 0.398443*a;

  basis0[8][0] = 0.898443*a;
  basis0[8][1] = 0.601557*a;
  basis0[8][2] = 0.398443*a;

  basis0[9][0] = 0.601557*a;
  basis0[9][1] = 0.601557*a;
  basis0[9][2] = 0.601557*a;

  basis0[10][0] = 0.601557*a;
  basis0[10][1] = 0.398443*a;
  basis0[10][2] = 0.898443*a;

  basis0[11][0] = 0.898443*a;
  basis0[11][1] = 0.398443*a;
  basis0[11][2] = 0.101557*a;

  basis0[12][0] = 0.101557*a;
  basis0[12][1] = 0.398443*a;
  basis0[12][2] = 0.601557*a;

  basis0[13][0] = 0.398443*a;
  basis0[13][1] = 0.398443*a;
  basis0[13][2] = 0.398443*a;

  basis0[14][0] = 0.898443*a;
  basis0[14][1] = 0.101557*a;
  basis0[14][2] = 0.601557*a;

  basis0[15][0] = 0.101557*a;
  basis0[15][1] = 0.601557*a;
  basis0[15][2] = 0.898443*a;
}

void ST12(const double scale, int &nb)
{
  double a, c;

  a = lc_a0*scale; //5.518891*Re*scale/2.26822; // volume per atom, c/a ratio and basis from Phys Rev B 49, 5329 (1994)
  c = 1.26*a;
  nb = 12;

  if (!basis0) basis0 = create(basis0,nb,3,"ST12:create()");
  else basis0 = grow(basis0,nb,3,"ST12:grow()");

  trans0[0][0] = a;
  trans0[0][1] = 0.0;
  trans0[0][2] = 0.0;

  trans0[1][0] = 0.0;
  trans0[1][1] = a;
  trans0[1][2] = 0.0;

  trans0[2][0] = 0.0;
  trans0[2][1] = 0.0;
  trans0[2][2] = c;

  basis0[0][0]=0.175200*a;
  basis0[0][1]=0.379200*a;
  basis0[0][2]=0.274200*c;
  basis0[1][0]=0.824700*a;
  basis0[1][1]=0.620700*a;
  basis0[1][2]=0.747200*c;
  basis0[2][0]=0.120800*a;
  basis0[2][1]=0.675000*a;
  basis0[2][2]=0.997300*c;
  basis0[3][0]=0.879200*a;
  basis0[3][1]=0.324800*a;
  basis0[3][2]=0.497300*c;
  basis0[4][0]=0.379200*a;
  basis0[4][1]=0.175200*a;
  basis0[4][2]=0.752800*c;
  basis0[5][0]=0.620700*a;
  basis0[5][1]=0.824700*a;
  basis0[5][2]=0.252700*c;
  basis0[6][0]=0.324800*a;
  basis0[6][1]=0.879100*a;
  basis0[6][2]=0.502700*c;
  basis0[7][0]=0.675100*a;
  basis0[7][1]=0.120800*a;
  basis0[7][2]=0.002700*c;
  basis0[8][0]=0.084900*a;
  basis0[8][1]=0.084900*a;
  basis0[8][2]=0.000000*c;
  basis0[9][0]=0.915100*a;
  basis0[9][1]=0.915100*a;
  basis0[9][2]=0.500000*c;
  basis0[10][0]=0.415100*a;
  basis0[10][1]=0.584800*a;
  basis0[10][2]=0.750000*c;
  basis0[11][0]=0.584900*a;
  basis0[11][1]=0.415100*a;
  basis0[11][2]=0.250000*c;
}

bool set_vac_supercell(const char *struc, const double scale, int &nb)
{
  double x0, y0, z0;
  bool ifound = false;
  double rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;

  if (strcmp(struc,"diam") == 0) {
    double a, a2, a4;
    int npbas = 8;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = true;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = a/2.0;
    a4 = a2/2.0;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    basis[4][0] = a4;
    basis[4][1] = a4;
    basis[4][2] = a4;

    basis[5][0] = 0.75*a;
    basis[5][1] = 0.75*a;
    basis[5][2] = 0.25*a;

    basis[6][0] = 0.75*a;
    basis[6][1] = 0.25*a;
    basis[6][2] = 0.75*a;

    basis[7][0] = 0.25*a;
    basis[7][1] = 0.75*a;
    basis[7][2] = 0.75*a;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_vac_supercell(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)-1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	    }
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("vac.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
    WriteConfigs("1vac","vac.dat",trans0,basis0,nb);
  }

  if (strcmp(struc,"fcc") == 0) {
    double a, a2;
    int npbas = 1;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = true;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = a/2.0;

    trans0[0][0] = a2;
    trans0[0][1] = a2;
    trans0[0][2] = 0.0;

    trans0[1][0] = a2;
    trans0[1][1] = 0.0;
    trans0[1][2] = a2;

    trans0[2][0] = 0.0;
    trans0[2][1] = a2;
    trans0[2][2] = a2;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    /*trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;*/

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_vac_supercell(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)-1;
    printf(" m1=%d m2=%d m3=%d nsite=%d\n",m1,m2,m3,nb);
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	    }
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
  }

  if (strcmp(struc,"bcc") == 0) {
    double a, a2;
    double basis[2][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = true;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = a/2.0;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = a2;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_vac_supercell(): %d %d %d\n",m1,m2,m3);
    nb = 2*(2*m1+1)*(2*m2+1)*(2*m3+1)-1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<2; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][0] + y0;
	      basis0[count][2] = basis[ii][0] + z0;
	      count++;
	    }
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
  }

  if (!ifound) {
    char message[1024];
    sprintf(message," given structure %s not found ...",struc);
    fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
    //if (nprocs > 1) MPI_Abort(world,errcode);
    //exit(EXIT_FAILURE);
  }

  return ifound;
}


bool set_Td_int(const char *struc, const double scale, int &nb)
{
  double x0, y0, z0;
  bool ifound = false;
  double rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;

  if (strcmp(struc,"diam") == 0) {
    double a, a2, a4;
    int npbas = 8;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = true;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;
    a4 = 0.25*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    basis[4][0] = a4;
    basis[4][1] = a4;
    basis[4][2] = a4;

    basis[5][0] = 0.75*a;
    basis[5][1] = 0.75*a;
    basis[5][2] = 0.25*a;

    basis[6][0] = 0.75*a;
    basis[6][1] = 0.25*a;
    basis[6][2] = 0.75*a;

    basis[7][0] = 0.25*a;
    basis[7][1] = 0.75*a;
    basis[7][2] = 0.75*a;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    // Add an interstitial at the tetrahedral site.
    basis0[count][0] = a2;
    basis0[count][1] = a2;
    basis0[count][2] = a2;

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("Td_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (strcmp(struc,"fcc") == 0) {
    double a, a2;
    int npbas = 4;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = true;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    // Add an interstitial at the tetrahedral site.
    basis0[count][0] = 0.25*a;
    basis0[count][1] = 0.25*a;
    basis0[count][2] = 0.25*a;

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("Td_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (!ifound) {
    char message[1024];
    sprintf(message," given structure %s not found ...",struc);
    fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
    //if (nprocs > 1) MPI_Abort(world,errcode);
    //exit(EXIT_FAILURE);
  }

  return ifound;
}

bool set_Octa_int(const char *struc, const double scale, int &nb)
{
  double x0, y0, z0;
  bool ifound = false;
  double rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;

  if (strcmp(struc,"fcc") == 0) {
    double a, a2;
    int npbas = 4;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = true;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    // Add an interstitial at the octahedral site.
    basis0[count][0] = a2;
    basis0[count][1] = a2;
    basis0[count][2] = a2;

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("Td_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (!ifound) {
    char message[1024];
    sprintf(message," given structure %s not found ...",struc);
    fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
    //if (nprocs > 1) MPI_Abort(world,errcode);
    //exit(EXIT_FAILURE);
  }

  return ifound;
}

bool set_HEX_int(const char *struc, const double scale, int &nb)
{
  double x0, y0, z0;
  bool ifound = false;
  double rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;

  if (strcmp(struc,"diam") == 0) {
    double a, a2, a4;
    int npbas = 8;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = true;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;
    a4 = 0.25*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    basis[4][0] = a4;
    basis[4][1] = a4;
    basis[4][2] = a4;

    basis[5][0] = 0.75*a;
    basis[5][1] = 0.75*a;
    basis[5][2] = 0.25*a;

    basis[6][0] = 0.75*a;
    basis[6][1] = 0.25*a;
    basis[6][2] = 0.75*a;

    basis[7][0] = 0.25*a;
    basis[7][1] = 0.75*a;
    basis[7][2] = 0.75*a;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    // Add an interstitial at a hexagonal site.
    basis0[count][0] = 3.0*a/8.0;
    basis0[count][1] = 3.0*a/8.0;
    basis0[count][2] = 5.0*a/8.0;

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("HEX_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (!ifound) {
    char message[1024];
    sprintf(message," given structure %s not found ...",struc);
    fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
    //if (nprocs > 1) MPI_Abort(world,errcode);
    //exit(EXIT_FAILURE);
  }

  return ifound;
}

bool set_B_int(const char *struc, const double scale, int &nb)
{
  double x0, y0, z0;
  bool ifound = false;
  double rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;

  if (strcmp(struc,"diam") == 0) {
    double a, a2, a4;
    int npbas = 8;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = true;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;
    a4 = 0.25*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    basis[4][0] = a4;
    basis[4][1] = a4;
    basis[4][2] = a4;

    basis[5][0] = 0.75*a;
    basis[5][1] = 0.75*a;
    basis[5][2] = 0.25*a;

    basis[6][0] = 0.75*a;
    basis[6][1] = 0.25*a;
    basis[6][2] = 0.75*a;

    basis[7][0] = 0.25*a;
    basis[7][1] = 0.75*a;
    basis[7][2] = 0.75*a;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }
    // Add an interstitial at a bond-center site.
    basis0[count][0] = a/8.0;
    basis0[count][1] = a/8.0;
    basis0[count][2] = a/8.0;

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("Bond_center_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (!ifound) {
    char message[1024];
    sprintf(message," given structure %s not found ...",struc);
    fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
    //if (nprocs > 1) MPI_Abort(world,errcode);
    //exit(EXIT_FAILURE);
  }

  return ifound;
}

void set_dumbbell110_int(const char *struc, const double scale, int &nb)
{
  double x0, y0, z0;
  int ifound = 0;
  double rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;

  if (strcmp(struc,"diam") == 0) {
    double a, a2, a4;
    int npbas = 8;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = 1;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;
    a4 = 0.25*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    basis[4][0] = a4;
    basis[4][1] = a4;
    basis[4][2] = a4;

    basis[5][0] = 0.75*a;
    basis[5][1] = 0.75*a;
    basis[5][2] = 0.25*a;

    basis[6][0] = 0.75*a;
    basis[6][1] = 0.25*a;
    basis[6][2] = 0.75*a;

    basis[7][0] = 0.25*a;
    basis[7][1] = 0.75*a;
    basis[7][2] = 0.75*a;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    bool found_site = false;
    // Find a site (a/4,a/4,a/4) and replace it with two interstitials.
    for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] < (a4+1.e-9) && basis0[i][2] > (a4-1.e-9))
	if (basis0[i][0] < (a4+1.e-9) && basis0[i][0] > (a4-1.e-9)
	    && basis0[i][1] < (a4+1.e-9) && basis0[i][1] > (a4-1.e-9)) {
	  found_site = true;
	  basis0[i][0] = a4 - 0.169*a;
	  basis0[i][1] = a4 - 0.169*a;
	  basis0[i][2] = a4;
	}
    // Add another interstitial site.
    basis0[count][0] = a4 + 0.169*a;
    basis0[count][1] = a4 + 0.169*a;
    basis0[count][2] = a4;

    // Find a site (a/2,a/2,0) and replace it with two interstitials.
    /*for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] == 0.0)
	if (basis0[i][0] < (a2+1.e-9) && basis0[i][0] > (a2-1.e-9)
	    && basis0[i][1] < (a2+1.e-9) && basis0[i][1] > (a2-1.e-9)) {
	  found_site = true;
	  basis0[i][0] = a2 - 0.169*a;
	  basis0[i][1] = a2 - 0.169*a;
	  basis0[i][2] = 0.0;
	}
    // Add another interstitial site.
    basis0[count][0] = a2 + 0.169*a;
    basis0[count][1] = a2 + 0.169*a;
    basis0[count][2] = 0.0;*/

    WriteConfigs("dumbbell110","dumbbell110_int.dat",trans0,basis0,nb);

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("dumbbell110_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (strcmp(struc,"fcc") == 0) {
    double a, a2;
    int npbas = 4;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = 1;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    double a4 = 0.25*a;
    bool found_site = false;
    // Find a site (0,0,0) and replace it with two interstitials.
    for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] < 1.e-6 && basis0[i][2] > -1.e-6)
	if (basis0[i][0] < 1.e-6 && basis0[i][0] > -1.e-6
	    && basis0[i][1] < 1.e-6 && basis0[i][1] > -1.e-6) {
	  found_site = true;
	  basis0[i][0] = a4;
	  basis0[i][1] = a4;
	  basis0[i][2] = 0.0;
	}
    // Add another interstitial site.
    basis0[count][0] = -a4;
    basis0[count][1] = -a4;
    basis0[count][2] = 0.0;

    // Find a site (a/2,a/2,0) and replace it with two interstitials.
    /*for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] == 0.0)
	if (basis0[i][0] < (a2+1.e-9) && basis0[i][0] > (a2-1.e-9)
	    && basis0[i][1] < (a2+1.e-9) && basis0[i][1] > (a2-1.e-9)) {
	  found_site = true;
	  basis0[i][0] = a2 - 0.169*a;
	  basis0[i][1] = a2 - 0.169*a;
	  basis0[i][2] = 0.0;
	}
    // Add another interstitial site.
    basis0[count][0] = a2 + 0.169*a;
    basis0[count][1] = a2 + 0.169*a;
    basis0[count][2] = 0.0;*/

    //WriteConfigs("dumbbell110","dumbbell110_int.dat",trans0,basis0,nb);

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("dumbbell110_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (!ifound) {
    char message[1024];
    sprintf(message," given structure %s not found ...",struc);
    fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }
}

void set_dumbbell100_int(const char *struc, const double scale, int &nb)
{
  double x0, y0, z0;
  int ifound = 0;
  double rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;

  if (strcmp(struc,"diam") == 0) {
    double a, a2, a4;
    int npbas = 8;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = 1;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;
    a4 = 0.25*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    basis[4][0] = a4;
    basis[4][1] = a4;
    basis[4][2] = a4;

    basis[5][0] = 0.75*a;
    basis[5][1] = 0.75*a;
    basis[5][2] = 0.25*a;

    basis[6][0] = 0.75*a;
    basis[6][1] = 0.25*a;
    basis[6][2] = 0.75*a;

    basis[7][0] = 0.25*a;
    basis[7][1] = 0.75*a;
    basis[7][2] = 0.75*a;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    bool found_site = false;
    // Find a site (0,0,0) and replace it with two interstitials.
    for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] < 1.e-6 && basis0[i][2] > -1.e-6)
	if (basis0[i][0] < 1.e-6 && basis0[i][0] > -1.e-6
	    && basis0[i][1] < 1.e-6 && basis0[i][1] > -1.e-6) {
	  found_site = true;
	  basis0[i][0] = 0.5*a*sqrt(3.0)/4.0;
	  basis0[i][1] = 0.0;
	  basis0[i][2] = 0.0;
	}
    // Add another interstitial site.
    basis0[count][0] = -0.5*a*sqrt(3.0)/4.0;
    basis0[count][1] = 0.0;
    basis0[count][2] = 0.0;

    // Find a site (a/2,a/2,0) and replace it with two interstitials.
    /*for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] == 0.0)
	if (basis0[i][0] < (a2+1.e-9) && basis0[i][0] > (a2-1.e-9)
	    && basis0[i][1] < (a2+1.e-9) && basis0[i][1] > (a2-1.e-9)) {
	  found_site = true;
	  basis0[i][0] = a2 - 0.169*a;
	  basis0[i][1] = a2 - 0.169*a;
	  basis0[i][2] = 0.0;
	}
    // Add another interstitial site.
    basis0[count][0] = a2 + 0.169*a;
    basis0[count][1] = a2 + 0.169*a;
    basis0[count][2] = 0.0;*/

    //WriteConfigs("dumbbell100","dumbbell100_int.dat",trans0,basis0,nb);

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("dumbbell100_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (strcmp(struc,"fcc") == 0) {
    double a, a2;
    int npbas = 4;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = 1;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    bool found_site = false;
    // Find a site (0,0,0) and replace it with two interstitials.
    for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] < 1.e-6 && basis0[i][2] > -1.e-6)
	if (basis0[i][0] < 1.e-6 && basis0[i][0] > -1.e-6
	    && basis0[i][1] < 1.e-6 && basis0[i][1] > -1.e-6) {
	  found_site = true;
	  basis0[i][0] = 0.5*a/sqrt(2.0);
	  basis0[i][1] = 0.0;
	  basis0[i][2] = 0.0;
	}
    // Add another interstitial site.
    basis0[count][0] = -0.5*a/sqrt(2.0);
    basis0[count][1] = 0.0;
    basis0[count][2] = 0.0;

    // Find a site (a/2,a/2,0) and replace it with two interstitials.
    /*for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] == 0.0)
	if (basis0[i][0] < (a2+1.e-9) && basis0[i][0] > (a2-1.e-9)
	    && basis0[i][1] < (a2+1.e-9) && basis0[i][1] > (a2-1.e-9)) {
	  found_site = true;
	  basis0[i][0] = a2 - 0.169*a;
	  basis0[i][1] = a2 - 0.169*a;
	  basis0[i][2] = 0.0;
	}
    // Add another interstitial site.
    basis0[count][0] = a2 + 0.169*a;
    basis0[count][1] = a2 + 0.169*a;
    basis0[count][2] = 0.0;*/

    //WriteConfigs("dumbbell100","dumbbell100_int.dat",trans0,basis0,nb);

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("dumbbell100_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (!ifound) {
    char message[1024];
    sprintf(message," given structure %s not found ...",struc);
    fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }
}

bool set_dumbbell_int(const char *struc, const char *type, const double scale, int &nb)
{
  double x0, y0, z0;
  bool ifound = false;
  double rc;

  rc = 1.5*Rc;
  if (PotentialType == 1) rc = Rc;

  if (strcmp(struc,"diam") == 0 && strcmp("111",type) != 0) {
    double a, a2, a4;
    int npbas = 8;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = true;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;
    a4 = 0.25*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    basis[4][0] = a4;
    basis[4][1] = a4;
    basis[4][2] = a4;

    basis[5][0] = 0.75*a;
    basis[5][1] = 0.75*a;
    basis[5][2] = 0.25*a;

    basis[6][0] = 0.75*a;
    basis[6][1] = 0.25*a;
    basis[6][2] = 0.75*a;

    basis[7][0] = 0.25*a;
    basis[7][1] = 0.75*a;
    basis[7][2] = 0.75*a;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    bool found_site = false;
    // Find a site (0,0,0) and replace it with two interstitials.
    for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] < 1.e-6 && basis0[i][2] > -1.e-6)
	if (basis0[i][0] < 1.e-6 && basis0[i][0] > -1.e-6
	    && basis0[i][1] < 1.e-6 && basis0[i][1] > -1.e-6) {
	  found_site = true;
	  if (strcmp("100",type) == 0) {
	    basis0[i][0] = -0.5*a*sqrt(3.0)/4.0;
	    basis0[i][1] = 0.0;
	    basis0[i][2] = 0.0;
	  }
	  if (strcmp("110",type) == 0) {
	    basis0[i][0] = -0.169*a;
	    basis0[i][1] = -0.169*a;
	    basis0[i][2] = 0.0;
	  }
	}
    // Add another interstitial site.
    if (strcmp("100",type) == 0) {
      basis0[count][0] = 0.5*a*sqrt(3.0)/4.0;
      basis0[count][1] = 0.0;
      basis0[count][2] = 0.0;
      //WriteConfigs("dumbbell100","dumbbell100_int.dat",trans0,basis0,nb);
    }
    if (strcmp("110",type) == 0) {
      basis0[count][0] = 0.169*a;
      basis0[count][1] = 0.169*a;
      basis0[count][2] = 0.0;
      //WriteConfigs("dumbbell110","dumbbell110_int.dat",trans0,basis0,nb);
    }

    // Find a site (a/2,a/2,0) and replace it with two interstitials.
    /*for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] == 0.0)
	if (basis0[i][0] < (a2+1.e-9) && basis0[i][0] > (a2-1.e-9)
	    && basis0[i][1] < (a2+1.e-9) && basis0[i][1] > (a2-1.e-9)) {
	  found_site = true;
	  basis0[i][0] = a2 - 0.169*a;
	  basis0[i][1] = a2 - 0.169*a;
	  basis0[i][2] = 0.0;
	}
    // Add another interstitial site.
    basis0[count][0] = a2 + 0.169*a;
    basis0[count][1] = a2 + 0.169*a;
    basis0[count][2] = 0.0;*/

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("dumbbell100_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (strcmp(struc,"fcc") == 0) {
    double a, a2;
    int npbas = 4;
    double basis[npbas][3];
    int m1,m2,m3;
    double dnorm[3];

    ifound = true;
    a = lc_a0*scale; // Re*4.0/sqrt(3.0)*scale;
    a2 = 0.5*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a;

    basis[0][0] = 0.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;

    basis[1][0] = a2;
    basis[1][1] = a2;
    basis[1][2] = 0.0;

    basis[2][0] = a2;
    basis[2][1] = 0.0;
    basis[2][2] = a2;

    basis[3][0] = 0.0;
    basis[3][1] = a2;
    basis[3][2] = a2;

    for (int m=0; m<3; m++) {
      dnorm[m] = sqrt(pow(trans0[m][0],2) +
	  pow(trans0[m][1],2) + pow(trans0[m][2],2));
    }
    //std::cout << "I am here ...\n";
    m1 = (int) (ceil(rc/dnorm[0])); /* number of times to replicate along X */
    m2 = (int) (ceil(rc/dnorm[1])); /* number of times to replicate along Y */
    m3 = (int) (ceil(rc/dnorm[2])); /* number of times to replicate along Z */

    //m1 = 2;
    //m2 = 2;
    //m3 = 2;
    //printf(" m1, m2 and m3 in set_Td_int(): %d %d %d\n",m1,m2,m3);
    nb = npbas*(2*m1+1)*(2*m2+1)*(2*m3+1)+1;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"set_vac_supercell:create()");
    else basis0 = grow(basis0,nb,3,"set_vec_supercell:grow");

    int count = 0;
    for (int ii=0; ii<npbas; ii++) {
      for (int i=-m1; i<=m1; i++) {
	for (int j=-m2; j<=m2; j++) {
	  for (int k=-m3; k<=m3; k++) {
	    x0 = trans0[0][0]*i + trans0[1][0]*j + trans0[2][0]*k;
	    y0 = trans0[0][1]*i + trans0[1][1]*j + trans0[2][1]*k;
	    z0 = trans0[0][2]*i + trans0[1][2]*j + trans0[2][2]*k;
	    //if (abs(i) || abs(j) || abs(k) || ii) {
	      basis0[count][0] = basis[ii][0] + x0;
	      basis0[count][1] = basis[ii][1] + y0;
	      basis0[count][2] = basis[ii][2] + z0;
	      count++;
	      //std::cout << nb << " " << count << "\n";
	    //}
	  }
	}
      }
    }

    // Update translation vectors;
    trans0[0][0] *= (2*m1+1);
    trans0[0][1] *= (2*m1+1);
    trans0[0][2] *= (2*m1+1);
    trans0[1][0] *= (2*m2+1);
    trans0[1][1] *= (2*m2+1);
    trans0[1][2] *= (2*m2+1);
    trans0[2][0] *= (2*m3+1);
    trans0[2][1] *= (2*m3+1);
    trans0[2][2] *= (2*m3+1);

    if (count != nb-1) {
      char message[1024];
      sprintf(message," supercell size %d not equal to specified no. of bases %d ...",count,nb);
      fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
      if (nprocs > 1) MPI_Abort(world,errcode);
      exit(EXIT_FAILURE);
    }

    bool found_site = false;
    // Find a site (0,0,0) and replace it with two interstitials.
    for (int i=0; i<nb && !found_site; i++)
      if (basis0[i][2] < 1.e-6 && basis0[i][2] > -1.e-6)
	if (basis0[i][0] < 1.e-6 && basis0[i][0] > -1.e-6
	    && basis0[i][1] < 1.e-6 && basis0[i][1] > -1.e-6) {
	  found_site = true;
	  if (strcmp("100",type) == 0) {
	    basis0[i][0] = 0.5*a/sqrt(2.0);
	    basis0[i][1] = 0.0;
	    basis0[i][2] = 0.0;
	  }
	  if (strcmp("110",type) == 0) {
	    basis0[i][0] = a/6.0;
	    basis0[i][1] = a/6.0;
	    basis0[i][2] = 0.0;
	  }
	  if (strcmp("111",type) == 0) {
	    basis0[i][0] = 0.5*a/sqrt(6.0);
	    basis0[i][1] = 0.5*a/sqrt(6.0);
	    basis0[i][2] = 0.5*a/sqrt(6.0);
	  }
	}
    // Add another interstitial site.
    if (strcmp("100",type) == 0) {
      basis0[count][0] = -0.5*a/sqrt(2.0);
      basis0[count][1] = 0.0;
      basis0[count][2] = 0.0;
      WriteConfigs("dumbbell100","dumbbell100_int.dat",trans0,basis0,nb);
      //WriteDump("dumbbell100.dump",trans0,basis0,nb,0);
    }
    if (strcmp("110",type) == 0) {
      basis0[count][0] = -a/6.0;
      basis0[count][1] = -a/6.0;
      basis0[count][2] = 0.0;
      WriteConfigs("dumbbell110","dumbbell110_int.dat",trans0,basis0,nb);
      //WriteDump("dumbbell110.dump",trans0,basis0,nb,0);
    }
    if (strcmp("111",type) == 0) {
      basis0[count][0] = -0.5*a/sqrt(6.0);
      basis0[count][1] = -0.5*a/sqrt(6.0);
      basis0[count][2] = -0.5*a/sqrt(6.0);
      WriteConfigs("dumbbell111","dumbbell111_int.dat",trans0,basis0,nb);
      //WriteDump("dumbbell111.dump",trans0,basis0,nb,0);
    }

    // Only for debugging purpose.
    /*FILE *out;
    out = fopen("dumbbell100_int.dump","w");
    fprintf(out,"ITEM: TIMESTEP\n");
    fprintf(out,"0\n");
    fprintf(out,"ITEM: NUMBER OF ATOMS\n");
    fprintf(out,"%d\n",nb);
    fprintf(out,"ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(out,"%f %f\n",0.0,trans0[0][0]);
    fprintf(out,"%f %f\n",0.0,trans0[1][1]);
    fprintf(out,"%f %f\n",0.0,trans0[2][2]);
    fprintf(out,"ITEM: ATOMS id type x y z\n");
    for (int i=0; i<nb; i++) fprintf(out,"%d %d %f %f %f\n",i+1,1,basis0[i][0],basis0[i][1],basis0[i][2]);
    fclose(out);*/
  }

  if (!ifound) {
    char message[1024];
    sprintf(message," given structure %s not found ...",struc);
    fprintf(stderr,"%s in file %s (function %s) at line %d ...\n",message,FERR);
    //if (nprocs > 1) MPI_Abort(world,errcode);
    //exit(EXIT_FAILURE);
  }

  return ifound;
}
