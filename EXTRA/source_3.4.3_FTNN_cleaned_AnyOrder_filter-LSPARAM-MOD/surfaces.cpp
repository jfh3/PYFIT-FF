#include <stdlib.h>
#include <math.h>
#include "globals.h"
#include "compute.h"
#include "surfaces.h"
#include "util.h"
#include "write.h"

void surf100()
{
  double E, Area;
  int nb, ifound = 0;

  max1 = 4;
  max2 = 4;
  max3 = 0;

  if (strcmp(basic_struc,"fcc") == 0) {
    ifound = 1;
    double a, a2, bb, bbb;
    a = lc_a0/sqrt(2.0); //Re;
    a2 = a/2.0;
    bb = a*sqrt(2.0);
    bbb = bb/2.0;
    nb = 6;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf100:create()");
    else basis0 = grow(basis0,nb,3,"surf100:grow()");

    Area = a*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = 3.0*bb;

    basis0[0][0] = 0.0;
    basis0[0][1] = 0.0;
    basis0[0][2] = 0.0;

    basis0[1][0] = a2;
    basis0[1][1] = a2;
    basis0[1][2] = bbb;

    basis0[2][0] = 0.0;
    basis0[2][1] = 0.0;
    basis0[2][2] = bb;

    basis0[3][0] = a2;
    basis0[3][1] = a2;
    basis0[3][2] = bbb + bb;

    basis0[4][0] = 0.0;
    basis0[4][1] = 0.0;
    basis0[4][2] = bb + bb;

    basis0[5][0] = a2;
    basis0[5][1] = a2;
    basis0[5][2] = bbb + bb + bb;
  }

  if (strcmp(basic_struc,"bcc") == 0) {
    ifound = 1;
    double a, a2;
    a = lc_a0; //2.0*Re/sqrt(3.0);
    a2 = a/2.0;
    nb = 6;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf100:create()");
    else basis0 = grow(basis0,nb,3,"surf100:grow()");

    Area = a*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = 3.0*a;

    basis0[0][0] = 0.0;
    basis0[0][1] = 0.0;
    basis0[0][2] = 0.0;

    basis0[1][0] = a2;
    basis0[1][1] = a2;
    basis0[1][2] = a2;

    basis0[2][0] = 0.0;
    basis0[2][1] = 0.0;
    basis0[2][2] = a;

    basis0[3][0] = a2;
    basis0[3][1] = a2;
    basis0[3][2] = a2 + a;

    basis0[4][0] = 0.0;
    basis0[4][1] = 0.0;
    basis0[4][2] = a + a;

    basis0[5][0] = a2;
    basis0[5][1] = a2;
    basis0[5][2] = a2 + a +a;
  }

  if (strcmp(basic_struc,"diam") == 0) {
    ifound = 1;
    double a,a2,bb,bbb;
    a = lc_a0/sqrt(2.0); //4.0*Re/sqrt(6.0);
    a2 = a/2.0;
    bb = a*sqrt(2.0);
    bbb = bb/2.0;
    nb = 12;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf100:create()");
    else basis0 = grow(basis0,nb,3,"surf100:grow()");

    Area = a*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = 3.0*bb;

    basis0[0][0] = 0.0;
    basis0[0][1] = 0.0;
    basis0[0][2] = 0.0;

    basis0[1][0] = 0.0;
    basis0[1][1] = a2;
    basis0[1][2] = bbb*0.5;

    basis0[2][0] = a2;
    basis0[2][1] = a2;
    basis0[2][2] = bbb;

    basis0[3][0] = a2;
    basis0[3][1] = a;
    basis0[3][2] = bbb*1.5;

    basis0[4][0] = 0.0;
    basis0[4][1] = 0.0;
    basis0[4][2] = bb;

    basis0[5][0] = 0.0;
    basis0[5][1] = a2;
    basis0[5][2] = bbb*0.5 + bb;

    basis0[6][0] = a2;
    basis0[6][1] = a2;
    basis0[6][2] = bbb + bb;

    basis0[7][0] = a2;
    basis0[7][1] = a;
    basis0[7][2] = bbb*1.5 + bb;

    basis0[8][0] = 0.0;
    basis0[8][1] = 0.0;
    basis0[8][2] = bb + bb;

    basis0[9][0] = 0.0;
    basis0[9][1] = a2;
    basis0[9][2] = bbb*0.5 + bb + bb;

    basis0[10][0] = a2;
    basis0[10][1] = a2;
    basis0[10][2] = bbb + bb + bb;

    basis0[11][0] = a2;
    basis0[11][1] = a;
    basis0[11][2] = bbb*1.5 + bb + bb;
  }

  if (ifound) {
    E = crystal_eng(trans0,basis0,nb);
    printf(" energy of (100) surface = %f (J/m^2)\n",(E-E0*nb)/Area/2.0*16.0219);
    // Only for debugging purpose.
    WriteConfigs("Surf100","surf100.dat",trans0,basis0,nb);
    //WriteDump("surf100.dump",trans0,basis0,nb,0);
  } else {
    char buf[256];
    sprintf(buf," cannot find structure: %s",basic_struc);
    errmsg(buf,FERR);
    //if (nprocs > 1) MPI_Abort(world,errcode);
    //exit(EXIT_FAILURE);
  }
  max1 = max2 = max3 = 1;
}

void surf110()
{
  double E, Area;
  int nb, ifound = 0;

  max1 = 4;
  max2 = 4;
  max3 = 0;

  if (strcmp(basic_struc,"fcc") == 0) {
    ifound = 1;
    double a,a2,bb,bbb;
    a = lc_a0/sqrt(2.0); //Re;
    a2 = a/2.0;
    bb = a*sqrt(2.0); // a0;
    bbb = bb/2.0; // a0/2;
    nb = 8;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf110:create()");
    else basis0 = grow(basis0,nb,3,"surf110:grow()");

    Area = a*bb;

    trans0[0][0] = bb;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = a*nb/2.0;

    basis0[0][0] = 0.0;
    basis0[0][1] = 0.0;
    basis0[0][2] = 0.0;

    basis0[1][0] = bbb;
    basis0[1][1] = a2;
    basis0[1][2] = a2;

    basis0[2][0] = 0.0;
    basis0[2][1] = 0.0;
    basis0[2][2] = a;

    basis0[3][0] = bbb;
    basis0[3][1] = a2;
    basis0[3][2] = a2 + a;

    basis0[4][0] = 0.0;
    basis0[4][1] = 0.0;
    basis0[4][2] = 2*a;

    basis0[5][0] = bbb;
    basis0[5][1] = a2;
    basis0[5][2] = a2 + 2*a;

    basis0[6][0] = 0.0;
    basis0[6][1] = 0.0;
    basis0[6][2] = 3*a;

    basis0[7][0] = bbb;
    basis0[7][1] = a2;
    basis0[7][2] = a2 + 3*a;
  }

  if (strcmp(basic_struc,"diam")==0) {
    ifound = 1;
    double a,a2,bb,bbb;
    a = lc_a0/sqrt(2.0); // 4.0*Re/sqrt(6.0);
    a2 = a/2.0;
    bb = a*sqrt(2.0);
    bbb = bb/2.0;
    nb = 20;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf110:create()");
    else basis0 = grow(basis0,nb,3,"surf110:grow()");

    Area = a*bb;

    trans0[0][0] = bb;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = 5.0*a;

    basis0[0][0] = bbb*0.5;
    basis0[0][1] = 0.0;
    basis0[0][2] = a2;

    basis0[1][0] = bbb;
    basis0[1][1] = a2;
    basis0[1][2] = a2;

    basis0[2][0] = bbb*1.5;
    basis0[2][1] = a2;
    basis0[2][2] = a;

    basis0[3][0] = 0.0;
    basis0[3][1] = 0.0;
    basis0[3][2] = a;

    basis0[4][0] = bbb*1.5;
    basis0[4][1] = a2;
    basis0[4][2] = a + a;

    basis0[5][0] = bbb;
    basis0[5][1] = a2;
    basis0[5][2] = a2 + a;

    basis0[6][0] = bbb*0.5;
    basis0[6][1] = 0.0;
    basis0[6][2] = a2 + a;

    basis0[7][0] = 0.0;
    basis0[7][1] = 0.0;
    basis0[7][2] = a + a;

    basis0[8][0] = bbb;
    basis0[8][1] = a2;
    basis0[8][2] = a2 + a + a;

    basis0[9][0] = bbb*0.5;
    basis0[9][1] = 0.0;
    basis0[9][2] = a2 + a + a;

    basis0[10][0] = bbb*1.5;
    basis0[10][1] = a2;
    basis0[10][2] = a + a + a;

    basis0[11][0] = 0.0;
    basis0[11][1] = 0.0;
    basis0[11][2] = a + a + a;

    basis0[12][0] = bbb;
    basis0[12][1] = a2;
    basis0[12][2] = a2 + a + a + a;

    basis0[13][0] = bbb*0.5;
    basis0[13][1] = 0.0;
    basis0[13][2] = a2 + a + a + a;

    basis0[14][0] = bbb*1.5;
    basis0[14][1] = a2;
    basis0[14][2] = a + a + a + a;

    basis0[15][0] = 0.0;
    basis0[15][1] = 0.0;
    basis0[15][2] = a + a + a + a;

    basis0[16][0] = bbb;
    basis0[16][1] = a2;
    basis0[16][2] = a2 + a + a + a + a;

    basis0[17][0] = bbb*0.5;
    basis0[17][1] = 0.0;
    basis0[17][2] = a2 + a + a + a + a;

    basis0[18][0] = bbb*1.5;
    basis0[18][1] = a2;
    basis0[18][2] = 5.0*a;

    basis0[19][0] = 0.0;
    basis0[19][1] = 0.0;
    basis0[19][2] = 5.0*a;

    /*
    //basis0[0][0] = 0.0;
    //basis0[0][1] = 0.0;
    //basis0[0][2] = 0.0;

    basis0[0][0] = bbb*0.5;
    basis0[0][1] = a2;
    basis0[0][2] = a;

    basis0[1][0] = bbb;
    basis0[1][1] = a2;
    basis0[1][2] = a2;

    basis0[2][0] = bbb*1.5;
    basis0[2][1] = a;
    basis0[2][2] = a2;

    basis0[3][0] = 0.0;
    basis0[3][1] = 0.0;
    basis0[3][2] = a;

    basis0[4][0] = bbb*0.5;
    basis0[4][1] = a2;
    basis0[4][2] = a + a;

    basis0[5][0] = bbb;
    basis0[5][1] = a2;
    basis0[5][2] = a2 + a;

    basis0[6][0] = bbb*1.5;
    basis0[6][1] = a;
    basis0[6][2] = a2 + a;

    basis0[7][0] = 0.0;
    basis0[7][1] = 0.0;
    basis0[7][2] = a + a;

    basis0[8][0] = bbb*0.5;
    basis0[8][1] = a2;
    basis0[8][2] = a + a + a;

    basis0[9][0] = bbb;
    basis0[9][1] = a2;
    basis0[9][2] = a2 + a + a;

    basis0[10][0] = bbb*1.5;
    basis0[10][1] = a;
    basis0[10][2] = a2 + a + a;

    basis0[11][0] = 0.0;
    basis0[11][1] = 0.0;
    basis0[11][2] = a + a + a;

    basis0[12][0] = bbb*0.5;
    basis0[12][1] = a2;
    basis0[12][2] = a + a + a + a;

    basis0[13][0] = bbb;
    basis0[13][1] = a2;
    basis0[13][2] = a2 + a + a + a;

    basis0[14][0] = bbb*1.5;
    basis0[14][1] = a;
    basis0[14][2] = a2 + a + a + a;

    basis0[15][0] = 0.0;
    basis0[15][1] = 0.0;
    basis0[15][2] = a + a + a + a;

    //basis0[12][0] = bbb*0.5;
    //basis0[12][1] = a2;
    //basis0[12][2] = a + a + a + a + a;

    basis0[16][0] = bbb;
    basis0[16][1] = a2;
    basis0[16][2] = a2 + a + a + a + a;

    basis0[17][0] = bbb*1.5;
    basis0[17][1] = a;
    basis0[17][2] = a2 + a + a + a + a;*/

  }

  if (strcmp(basic_struc,"bcc") == 0) {
    ifound = 1;
    double a,a2,bb,bbb,b3;
    a = lc_a0; // 2.0*Re/sqrt(3.0);
    a2 = a/2.0;
    bb = a*sqrt(2.0);
    bbb = bb/2.0;
    b3 = 3.0*bbb;
    nb = 6;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf110:create()");
    else basis0 = grow(basis0,nb,3,"surf110:grow()");
    Area = a*bb;
    trans0[0][0] = bb;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = b3;
    trans0[2][1] = 0.0;
    trans0[2][2] = b3;

    basis0[0][0] = 0.0;
    basis0[0][1] = 0.0;
    basis0[0][2] = 0.0;

    basis0[1][0] = bbb;
    basis0[1][1] = a2;
    basis0[1][2] = 0.0;

    basis0[2][0] = bbb;
    basis0[2][1] = 0.0;
    basis0[2][2] = bbb;

    basis0[3][0] = bb;
    basis0[3][1] = a2;
    basis0[3][2] = bbb;

    basis0[4][0] = bb;
    basis0[4][1] = 0.0;
    basis0[4][2] = bb;

    basis0[5][0] = bbb + bb;
    basis0[5][1] = a2;
    basis0[5][2] = bb;
  }

  if (ifound) {
    E = crystal_eng(trans0,basis0,nb);
    printf(" energy of (110) surface = %f (J/m^2)\n",(E-E0*nb)/Area/2.0*16.0219);
    // Only for debugging purpose.
    WriteConfigs("Surf110","surf110.dat",trans0,basis0,nb);
    //WriteDump("surf110.dump",trans0,basis0,nb,0);
  } else {
    char buf[256];
    sprintf(buf," cannot find structure: %s",basic_struc);
    errmsg(buf,FERR);
    //if (nprocs > 1) MPI_Abort(world,errcode);
    //exit(EXIT_FAILURE);
  }
  max1 = max2 = max3 = 1;
}

void surf111()
{
  double E, Area;
  int nb, ifound = 0;

  max1 = 4;
  max2 = 4;
  max3 = 0;

  if (strcmp(basic_struc,"fcc") == 0) {
    ifound = 1;
    double a,a2,bb,bbb,c,cc;
    a = lc_a0/sqrt(2.0); // Re;
    a2 = a/2.0;
    bb = a*sqrt(3.0)/2.0;
    bbb = bb/3.0;
    c =  2.0*lc_a0/sqrt(3.0); // Re*sqrt(8.0/3.0);
    nb = 9;
    cc = nb*c/2.0;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf111:create()");
    else basis0 = grow(basis0,nb,3,"surf111:grow()");
    Area = a*bb;
    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = a2;
    trans0[1][1] = bb;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = cc;

    basis0[0][0] = 0.0;    // A
    basis0[0][1] = 0.0;    // A
    basis0[0][2] = 0.0;    // A

    basis0[1][0] = a2;     // B
    basis0[1][1] = bbb;    // B
    basis0[1][2] = c/2.0;  // B

    basis0[2][0] = a2;      // C
    basis0[2][1] = -bbb;    // C
    basis0[2][2] = c;       // C

    basis0[3][0] = 0.0;     // A
    basis0[3][1] = 0.0;     // A
    basis0[3][2] = 1.50*c;  // A

    basis0[4][0] = a2;       // B
    basis0[4][1] = bbb;      // B
    basis0[4][2] = 2.0*c;    // B

    basis0[5][0] = a2;       // C
    basis0[5][1] = -bbb;     // C
    basis0[5][2] = 2.50*c;   // C

    basis0[6][0] = 0.0;     // A
    basis0[6][1] = 0.0;     // A
    basis0[6][2] = 3.0*c;   // A

    basis0[7][0] = a2;       // B
    basis0[7][1] = bbb;      // B
    basis0[7][2] = 3.50*c;   // B

    basis0[8][0] = a2;       // C
    basis0[8][1] = -bbb;     // C
    basis0[8][2] = 4.0*c;    // C
  }

  if (strcmp(basic_struc,"diam")==0) {
    ifound = 1;
    double a,aa,a2,bb,bbb,c,cc;
    aa = lc_a0*sqrt(3.0)/4.0; // Re == nearest neighbor in diamond structure;
    a = lc_a0/sqrt(2.0); // 4.0*Re/sqrt(6.0);
    a2 = a/2.0;
    bb = a*sqrt(3.0)/2.0;
    c =  a*sqrt(8.0/3.0);
    bbb = bb/3.0;
    nb = 16;
    cc = nb*c/4;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf111:create()");
    else basis0 = grow(basis0,nb,3,"surf111:grow()");
    Area = a*bb;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = a2;
    trans0[1][1] = bb;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = cc;

    // (111) surfaces exposed along a glide plane.
    // each surface atom has three dangling atoms.
    /*
    basis0[0][0] = 0.0;    // A
    basis0[0][1] = 0.0;    // A
    basis0[0][2] = 0.0;    // A

    basis0[1][0] = a2;     // B
    basis0[1][1] = bbb;    // B
    basis0[1][2] = c/2.0;  // B

    basis0[2][0] = a2;      // C
    basis0[2][1] = -bbb;    // C
    basis0[2][2] = c;       // C

    basis0[3][0] = 0.0;     // A
    basis0[3][1] = 0.0;     // A
    basis0[3][2] = 1.50*c;  // A

    basis0[4][0] = a2;       // B
    basis0[4][1] = bbb;      // B
    basis0[4][2] = 2.0*c;    // B

    basis0[5][0] = a2;       // C
    basis0[5][1] = -bbb;     // C
    basis0[5][2] = 2.50*c;   // C

    basis0[6][0] = 0.0;     // A
    basis0[6][1] = 0.0;     // A
    basis0[6][2] = 3.0*c;   // A

    basis0[7][0] = a2;       // B
    basis0[7][1] = bbb;      // B
    basis0[7][2] = 3.50*c;   // B

    basis0[8][0] = a2;       // C
    basis0[8][1] = -bbb;     // C
    basis0[8][2] = 4.0*c;    // C

    basis0[9][0] = 0.0;    // A'
    basis0[9][1] = 0.0;    // A'
    basis0[9][2] = aa;      // A'

    basis0[10][0] = a2;         // B'
    basis0[10][1] = bbb;        // B'
    basis0[10][2] = c/2.0 + aa;  // B'

    basis0[11][0] = a2;          // C'
    basis0[11][1] = -bbb;        // C'
    basis0[11][2] = c + aa;       // C'

    basis0[12][0] = 0.0;          // A'
    basis0[12][1] = 0.0;          // A'
    basis0[12][2] = 1.50*c + aa;  // A'

    basis0[13][0] = a2;            // B'
    basis0[13][1] = bbb;           // B'
    basis0[13][2] = 2.0*c + aa;    // B'

    basis0[14][0] = a2;            // C'
    basis0[14][1] = -bbb;          // C'
    basis0[14][2] = 2.50*c + aa;   // C'

    basis0[15][0] = 0.0;          // A'
    basis0[15][1] = 0.0;          // A'
    basis0[15][2] = 3.0*c + aa;   // A'

    basis0[16][0] = a2;           // B'
    basis0[16][1] = bbb;          // B'
    basis0[16][2] = 3.50*c + aa;  // B'

    basis0[17][0] = a2;           // C'
    basis0[17][1] = -bbb;         // C'
    basis0[17][2] = 4.0*c + aa;   // C'*/

    // (111) surfaces exposed along a suffle plane.
    // each surface atom has one dangling bond.
    //basis0[0][0] = 0.0;    // A
    //basis0[0][1] = 0.0;    // A
    //basis0[0][2] = 0.0;    // A

    basis0[0][0] = a2;     // B
    basis0[0][1] = bbb;    // B
    basis0[0][2] = c/2.0;  // B

    basis0[1][0] = a2;      // C
    basis0[1][1] = -bbb;    // C
    basis0[1][2] = c;       // C

    basis0[2][0] = 0.0;     // A
    basis0[2][1] = 0.0;     // A
    basis0[2][2] = 1.50*c;  // A

    basis0[3][0] = a2;       // B
    basis0[3][1] = bbb;      // B
    basis0[3][2] = 2.0*c;    // B

    basis0[4][0] = a2;       // C
    basis0[4][1] = -bbb;     // C
    basis0[4][2] = 2.50*c;   // C

    basis0[5][0] = 0.0;     // A
    basis0[5][1] = 0.0;     // A
    basis0[5][2] = 3.0*c;   // A

    basis0[6][0] = a2;       // B
    basis0[6][1] = bbb;      // B
    basis0[6][2] = 3.50*c;   // B

    basis0[7][0] = a2;       // C
    basis0[7][1] = -bbb;     // C
    basis0[7][2] = 4.0*c;    // C

    basis0[8][0] = 0.0;    // A'
    basis0[8][1] = 0.0;    // A'
    basis0[8][2] = aa;      // A'

    basis0[9][0] = a2;         // B'
    basis0[9][1] = bbb;        // B'
    basis0[9][2] = c/2.0 + aa;  // B'

    basis0[10][0] = a2;          // C'
    basis0[10][1] = -bbb;        // C'
    basis0[10][2] = c + aa;       // C'

    basis0[11][0] = 0.0;          // A'
    basis0[11][1] = 0.0;          // A'
    basis0[11][2] = 1.50*c + aa;  // A'

    basis0[12][0] = a2;            // B'
    basis0[12][1] = bbb;           // B'
    basis0[12][2] = 2.0*c + aa;    // B'

    basis0[13][0] = a2;            // C'
    basis0[13][1] = -bbb;          // C'
    basis0[13][2] = 2.50*c + aa;   // C'

    basis0[14][0] = 0.0;          // A'
    basis0[14][1] = 0.0;          // A'
    basis0[14][2] = 3.0*c + aa;   // A'

    basis0[15][0] = a2;           // B'
    basis0[15][1] = bbb;          // B'
    basis0[15][2] = 3.50*c + aa;  // B'

    //basis0[16][0] = a2;           // C'
    //basis0[16][1] = -bbb;         // C'
    //basis0[16][2] = 4.0*c + aa;   // C'
  }

  if (strcmp(basic_struc,"bcc")==0) {
    ifound = 1;
    double a,a2,bb,bbb,c,cc;
    a = lc_a0*sqrt(2.0); // Re*sqrt(8.0/3.0);
    a2 = a/2.0;
    bb = a*sqrt(3.0)/2.0;
    bbb = bb/3.0;
    c =  lc_a0/sqrt(3.0); // 2.0*Re/3.0;
    Area = a*bb;
    nb = 9;
    cc = nb*c;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf111:create()");
    else basis0 = grow(basis0,nb,3,"surf111:grow()");

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = a2;
    trans0[1][1] = bb;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = cc;

    basis0[0][0] = 0.0;     // A
    basis0[0][1] = 0.0;     // A
    basis0[0][2] = 0.0;     // A

    basis0[1][0] = a2;       // B
    basis0[1][1] = bbb;      // B
    basis0[1][2] = c/2.0;   // B

    basis0[2][0] = a2;       // C
    basis0[2][1] = -bbb;     // C
    basis0[2][2] = c;        // C

    basis0[3][0] = 0.0;     // A
    basis0[3][1] = 0.0;     // A
    basis0[3][2] = 1.50*c;  // A

    basis0[4][0] = a2;       // B
    basis0[4][1] = bbb;      // B
    basis0[4][2] = 2.0*c;   // B

    basis0[5][0] = a2;       // C
    basis0[5][1] = -bbb;     // C
    basis0[5][2] = 2.50*c;  // C

    basis0[6][0] = 0.0;     // A
    basis0[6][1] = 0.0;     // A
    basis0[6][2] = 3.0*c;   // A

    basis0[7][0] = a2;       // B
    basis0[7][1] = bbb;      // B
    basis0[7][2] = 3.50*c;  // B

    basis0[8][0] = a2;       // C
    basis0[8][1] = -bbb;     // C
    basis0[8][2] = 4.0*c;   // C
  }

  if (ifound) {
    E = crystal_eng(trans0,basis0,nb);
    printf(" energy of (111) surface = %f (J/m^2)\n",(E-E0*nb)/Area/2.0*16.0219);
    // Only for debugging purpose.
    WriteConfigs("Surf111","surf111.dat",trans0,basis0,nb);
    //WriteDump("surf111.dump",trans0,basis0,nb,0);
  } else {
    char buf[256];
    sprintf(buf," cannot find structure: %s",basic_struc);
    errmsg(buf,FERR);
    //if (nprocs > 1) MPI_Abort(world,errcode);
    //exit(EXIT_FAILURE);
  }
  max1 = max2 = max3 = 1;
}


void SFp111()
{
  double Area;
  int nb, ifound = 0;

  max1 = 4;
  max2 = 4;
  max3 = 0;

  if (strcmp(basic_struc,"bcc")==0) {
    ifound = 1;
    double a,a2,bb,bbb,c,cc;
    a = lc_a0*sqrt(2.0); // Re*sqrt(8.0/3.0);
    a2 = a/2.0;
    bb = a*sqrt(3.0)/2.0;
    bbb = bb/3.0;
    c =  lc_a0/sqrt(3.0); // 2.0*Re/3.0;
    Area = a*bb;
    nb = 9;
    cc = nb*c;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf111:create()");
    else basis0 = grow(basis0,nb,3,"surf111:grow()");

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = a2;
    trans0[1][1] = bb;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = cc;

    basis0[0][0] = 0.0;     // A
    basis0[0][1] = 0.0;     // A
    basis0[0][2] = 0.0;     // A

    basis0[1][0] = a2;       // B
    basis0[1][1] = bbb;      // B
    basis0[1][2] = c/2.0;   // B

    basis0[2][0] = a2;       // C
    basis0[2][1] = -bbb;     // C
    basis0[2][2] = c;        // C

    basis0[3][0] = 0.0;     // A
    basis0[3][1] = 0.0;     // A
    basis0[3][2] = 1.50*c;  // A

    basis0[4][0] = a2;       // B
    basis0[4][1] = bbb;      // B
    basis0[4][2] = 2.0*c;   // B

    basis0[5][0] = a2;       // C
    basis0[5][1] = -bbb;     // C
    basis0[5][2] = 2.50*c;  // C

    basis0[6][0] = 0.0;     // A
    basis0[6][1] = 0.0;     // A
    basis0[6][2] = 3.0*c;   // A

    basis0[7][0] = a2;       // B
    basis0[7][1] = bbb;      // B
    basis0[7][2] = 3.50*c;  // B

    basis0[8][0] = a2;       // C
    basis0[8][1] = -bbb;     // C
    basis0[8][2] = 4.0*c;   // C
  }

  if (strcmp(basic_struc,"fcc") == 0) {
    ifound = 1;
    double a,a2,bb,bbb,c,cc;
    a = lc_a0/sqrt(2.0); // Re;
    a2 = a/2.0;
    bb = a*sqrt(3.0)/2.0;
    bbb = bb/3.0;
    c =  2.0*lc_a0/sqrt(3.0); // Re*sqrt(8.0/3.0);
    nb = 9;
    cc = nb*c;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf111:create()");
    else basis0 = grow(basis0,nb,3,"surf111:grow()");
    Area = a*bb;
    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = a2;
    trans0[1][1] = bb;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = cc;

    basis0[0][0] = 0.0;    // A
    basis0[0][1] = 0.0;    // A
    basis0[0][2] = 0.0;    // A

    basis0[1][0] = a2;     // B
    basis0[1][1] = bbb;    // B
    basis0[1][2] = c/2.0;  // B

    basis0[2][0] = a2;      // C
    basis0[2][1] = -bbb;    // C
    basis0[2][2] = c;       // C

    basis0[3][0] = 0.0;     // A
    basis0[3][1] = 0.0;     // A
    basis0[3][2] = 1.50*c;  // A

    basis0[4][0] = a2;       // B
    basis0[4][1] = bbb;      // B
    basis0[4][2] = 2.0*c;    // B

    basis0[5][0] = a2;       // C
    basis0[5][1] = -bbb;     // C
    basis0[5][2] = 2.50*c;   // C

    basis0[6][0] = 0.0;     // A
    basis0[6][1] = 0.0;     // A
    basis0[6][2] = 3.0*c;   // A

    basis0[7][0] = a2;       // B
    basis0[7][1] = bbb;      // B
    basis0[7][2] = 3.50*c;   // B

    basis0[8][0] = a2;       // C
    basis0[8][1] = -bbb;     // C
    basis0[8][2] = 4.0*c;    // C
  }

  if (strcmp(basic_struc,"diam")==0) {
    ifound = 1;
    double a,aa,a2,bb,bbb,c,cc;
    aa = lc_a0*sqrt(3.0)/4.0; // Re == nearest neighbor in diamond structure;
    a = lc_a0/sqrt(2.0); // 4.0*Re/sqrt(6.0);
    a2 = a/2.0;
    bb = a*sqrt(3.0)/2.0;
    c =  a*sqrt(8.0/3.0);
    bbb = bb/3.0;
    nb = 18;
    cc = nb*c/4;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf111:create()");
    else basis0 = grow(basis0,nb,3,"surf111:grow()");
    Area = a*bb;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = a2;
    trans0[1][1] = bb;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = cc;

    basis0[0][0] = 0.0;    // A
    basis0[0][1] = 0.0;    // A
    basis0[0][2] = 0.0;    // A

    basis0[1][0] = a2;     // B
    basis0[1][1] = bbb;    // B
    basis0[1][2] = c/2.0;  // B

    basis0[2][0] = a2;      // C
    basis0[2][1] = -bbb;    // C
    basis0[2][2] = c;       // C

    basis0[3][0] = 0.0;     // A
    basis0[3][1] = 0.0;     // A
    basis0[3][2] = 1.50*c;  // A

    basis0[4][0] = a2;       // B
    basis0[4][1] = bbb;      // B
    basis0[4][2] = 2.0*c;    // B

    basis0[5][0] = a2;       // C
    basis0[5][1] = -bbb;     // C
    basis0[5][2] = 2.50*c;   // C

    basis0[6][0] = 0.0;     // A
    basis0[6][1] = 0.0;     // A
    basis0[6][2] = 3.0*c;   // A

    basis0[7][0] = a2;       // B
    basis0[7][1] = bbb;      // B
    basis0[7][2] = 3.50*c;   // B

    basis0[8][0] = a2;       // C
    basis0[8][1] = -bbb;     // C
    basis0[8][2] = 4.0*c;    // C

    basis0[9][0] = 0.0;    // A'
    basis0[9][1] = 0.0;    // A'
    basis0[9][2] = aa;      // A'

    basis0[10][0] = a2;         // B'
    basis0[10][1] = bbb;        // B'
    basis0[10][2] = c/2.0 + aa;  // B'

    basis0[11][0] = a2;          // C'
    basis0[11][1] = -bbb;        // C'
    basis0[11][2] = c + aa;       // C'

    basis0[12][0] = 0.0;          // A'
    basis0[12][1] = 0.0;          // A'
    basis0[12][2] = 1.50*c + aa;  // A'

    basis0[13][0] = a2;            // B'
    basis0[13][1] = bbb;           // B'
    basis0[13][2] = 2.0*c + aa;    // B'

    basis0[14][0] = a2;            // C'
    basis0[14][1] = -bbb;          // C'
    basis0[14][2] = 2.50*c + aa;   // C'

    basis0[15][0] = 0.0;          // A'
    basis0[15][1] = 0.0;          // A'
    basis0[15][2] = 3.0*c + aa;   // A'

    basis0[16][0] = a2;           // B'
    basis0[16][1] = bbb;          // B'
    basis0[16][2] = 3.50*c + aa;  // B'

    basis0[17][0] = a2;           // C'
    basis0[17][1] = -bbb;         // C'
    basis0[17][2] = 4.0*c + aa;   // C'
  }

  if (ifound) {
    double **basis;
    double E00, E, c;
    int nds = 50;
    //double ds;
    FILE *out;

    basis = create(basis,nb,3,"SFp111_d211:create()");
    c = lc_a0*sqrt(4.0/3.0);
    E00 = crystal_eng(trans0,basis0,nb);
    // <211> direction on glide (111) plane.
    /*c = lc_a0*sqrt(4.0/3.0);
    ds = lc_a0*sqrt(3.0/2.0)/nds;
    if (strcmp(basic_struc,"bcc") == 0) {
      c = lc_a0/sqrt(3.0);
      ds = lc_a0*sqrt(6.0);
    }
    out = fopen("SFg111_d211.eng","w");
    E00 = crystal_eng(trans0,basis0,nb);
    for (int i=0; i<=nds; i++) {
      for (int j=0; j<nb; j++) {
	if (basis0[j][2]>2.0*c-0.01) {
	  basis[j][0] = basis0[j][0];
	  basis[j][1] = basis0[j][1] + i*ds;
	  basis[j][2] = basis0[j][2];
	} else {
	  basis[j][0] = basis0[j][0];
	  basis[j][1] = basis0[j][1];
	  basis[j][2] = basis0[j][2];
	}
      }
      if (i == 85) WriteDump("SFg111_d211.dump",trans0,basis,nb,i);
      E = crystal_eng(trans0,basis,nb);
      fprintf(out ," %f %f\n",i*ds,(E-E00)*16.0218/Area);
    }
    fclose(out);
    //WriteDump("SFg111_d211.dump",trans0,basis0,nb);

    // <110> direction on glide (111) plane.
    out = fopen("SFg111_d110.eng","w");
    ds = lc_a0/sqrt(2.0)/nds;
    if (strcmp(basic_struc,"bcc") == 0) ds = lc_a0*sqrt(2.0);
    for (int i=0; i<=nds; i++) {
      for (int j=0; j<nb; j++) {
	if (basis0[j][2]>2.0*c-0.01) {
	  basis[j][0] = basis0[j][0] + i*ds;
	  basis[j][1] = basis0[j][1];
	  basis[j][2] = basis0[j][2];
	} else {
	  basis[j][0] = basis0[j][0];
	  basis[j][1] = basis0[j][1];
	  basis[j][2] = basis0[j][2];
	}
      }
      if (i == 85) WriteDump("SFg111_d110.dump",trans0,basis,nb,i);
      E = crystal_eng(trans0,basis,nb);
      fprintf(out ," %f %f\n",i*ds,(E-E00)*16.0218/Area);
    }
    fclose(out);*/

    out = fopen("SFg111.eng","w");
    double dsx, dsy;
    dsx = lc_a0/sqrt(2.0)/nds;
    dsy = lc_a0*sqrt(3.0/2.0)/nds;
    for (int i=0; i<=nds; i++) {
      for (int j=0; j<=nds; j++) {
	for (int k=0; k<nb; k++) {
	  if (basis0[k][2]>2.0*c-0.01) {
	    basis[k][0] = basis0[k][0] + i*dsx; // <110> direction.
	    basis[k][1] = basis0[k][1] + j*dsy; // <211> direction.
	    basis[k][2] = basis0[k][2];
	  } else {
	    basis[k][0] = basis0[k][0];
	    basis[k][1] = basis0[k][1];
	    basis[k][2] = basis0[k][2];
	  }
	}
	WriteConfigs("SFg111","SFg111.dat",trans0,basis0,nb);
	//WriteDump("SFg111.dump",trans0,basis,nb,i*nds+j);
	E = crystal_eng(trans0,basis,nb);
	fprintf(out ," %f %f %f\n",i*dsx,j*dsy,(E-E00)*16.0218/Area);
      }
    }
    fclose(out);

    if (strcmp(basic_struc,"diam") == 0) {
      // For <110> direction on shuffle (111) plane.
      double aa = lc_a0*sqrt(3.0)/4.0;
      /*out = fopen("SFs111_d110.eng","w");
      for (int i=0; i<=nds; i++) {
	for (int j=0; j<nb; j++) {
	  if (basis0[j][2]>2.0*c+aa-0.01) {
	    basis[j][0] = basis0[j][0] + i*dsx; // <110> direction.
	    basis[j][1] = basis0[j][1];
	    basis[j][2] = basis0[j][2];
	  } else {
	    basis[j][0] = basis0[j][0];
	    basis[j][1] = basis0[j][1];
	    basis[j][2] = basis0[j][2];
	  }
	}
	//WriteDump("SFs111_d110.dump",trans0,basis,nb,i);
	E = crystal_eng(trans0,basis,nb);
	fprintf(out ," %f %f\n",i*dsx,(E-E00)*16.0218/Area);
      }
      fclose(out);*/

      // complete surface scan.
      out = fopen("SFs111.eng","w");
      for (int i=0; i<=nds; i++) {
	for (int j=0; j<=nds; j++) {
	  for (int k=0; k<nb; k++) {
	    if (basis0[k][2]>2.0*c+aa-0.01) { // aa declared above.
	      basis[k][0] = basis0[k][0] + i*dsx; // <110> direction.
	      basis[k][1] = basis0[k][1] + j*dsy; // <211> direction.
	      basis[k][2] = basis0[k][2];
	    } else {
	      basis[k][0] = basis0[k][0];
	      basis[k][1] = basis0[k][1];
	      basis[k][2] = basis0[k][2];
	    }
	  }
	  WriteConfigs("SFs111","SFs111.dat",trans0,basis0,nb);
	  //WriteDump("SFs111.dump",trans0,basis,nb,i*nds+j);
	  E = crystal_eng(trans0,basis,nb);
	  fprintf(out ," %f %f %f\n",i*dsx,j*dsy,(E-E00)*16.0218/Area);
	}
      }
      fclose(out);
    }
    destroy(basis);
  } else {
    char buf[256];
    sprintf(buf," cannot find structure: %s",basic_struc);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }
  max1 = max2 = max3 = 1;
}


void SFp100()
{
  double Area;
  int nb, ifound = 0;

  max1 = 4;
  max2 = 4;
  max3 = 0;

  if (strcmp(basic_struc,"fcc") == 0) {
    ifound = 1;
    double a, a2, bb, bbb;
    a = lc_a0/sqrt(2.0); //Re;
    a2 = a/2.0;
    bb = a*sqrt(2.0);
    bbb = bb/2.0;
    nb = 6;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf100:create()");
    else basis0 = grow(basis0,nb,3,"surf100:grow()");

    Area = a*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = 3.0*bb;

    basis0[0][0] = 0.0;
    basis0[0][1] = 0.0;
    basis0[0][2] = 0.0;

    basis0[1][0] = a2;
    basis0[1][1] = a2;
    basis0[1][2] = bbb;

    basis0[2][0] = 0.0;
    basis0[2][1] = 0.0;
    basis0[2][2] = bb;

    basis0[3][0] = a2;
    basis0[3][1] = a2;
    basis0[3][2] = bbb + bb;

    basis0[4][0] = 0.0;
    basis0[4][1] = 0.0;
    basis0[4][2] = bb + bb;

    basis0[5][0] = a2;
    basis0[5][1] = a2;
    basis0[5][2] = bbb + bb + bb;
  }

  if (strcmp(basic_struc,"bcc") == 0) {
    ifound = 1;
    double a, a2;
    a = lc_a0; //2.0*Re/sqrt(3.0);
    a2 = a/2.0;
    nb = 10;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf100:create()");
    else basis0 = grow(basis0,nb,3,"surf100:grow()");

    Area = a*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = 3.0*a;

    basis0[0][0] = 0.0;
    basis0[0][1] = 0.0;
    basis0[0][2] = 0.0;

    basis0[1][0] = a2;
    basis0[1][1] = a2;
    basis0[1][2] = a2;

    basis0[2][0] = 0.0;
    basis0[2][1] = 0.0;
    basis0[2][2] = a;

    basis0[3][0] = a2;
    basis0[3][1] = a2;
    basis0[3][2] = a2 + a;

    basis0[4][0] = 0.0;
    basis0[4][1] = 0.0;
    basis0[4][2] = a + a;

    basis0[5][0] = a2;
    basis0[5][1] = a2;
    basis0[5][2] = a2 + a +a;

    basis0[6][0] = 0.0;
    basis0[6][1] = 0.0;
    basis0[6][2] = 3.0*a;

    basis0[7][0] = a2;
    basis0[7][1] = a2;
    basis0[7][2] = a2 + 3.0*a;

    basis0[8][0] = 0.0;
    basis0[8][1] = 0.0;
    basis0[8][2] = 4.0*a;

    basis0[9][0] = a2;
    basis0[9][1] = a2;
    basis0[9][2] = a2 + 4.0*a;
  }

  if (strcmp(basic_struc,"diam") == 0) {
    ifound = 1;
    double a,a2,bb,bbb;
    a = lc_a0/sqrt(2.0); //4.0*Re/sqrt(6.0);
    a2 = a/2.0;
    bb = a*sqrt(2.0);
    bbb = bb/2.0;
    nb = 12;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf100:create()");
    else basis0 = grow(basis0,nb,3,"surf100:grow()");

    Area = a*a;

    trans0[0][0] = a;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = 3.0*bb;

    basis0[0][0] = 0.0;
    basis0[0][1] = 0.0;
    basis0[0][2] = 0.0;

    basis0[1][0] = 0.0;
    basis0[1][1] = a2;
    basis0[1][2] = bbb*0.5;

    basis0[2][0] = a2;
    basis0[2][1] = a2;
    basis0[2][2] = bbb;

    basis0[3][0] = a2;
    basis0[3][1] = a;
    basis0[3][2] = bbb*1.5;

    basis0[4][0] = 0.0;
    basis0[4][1] = 0.0;
    basis0[4][2] = bb;
    // --------- glide plane --------
    basis0[5][0] = 0.0;
    basis0[5][1] = a2;
    basis0[5][2] = bbb*0.5 + bb;

    basis0[6][0] = a2;
    basis0[6][1] = a2;
    basis0[6][2] = bbb + bb;

    basis0[7][0] = a2;
    basis0[7][1] = a;
    basis0[7][2] = bbb*1.5 + bb;

    basis0[8][0] = 0.0;
    basis0[8][1] = 0.0;
    basis0[8][2] = bb + bb;

    basis0[9][0] = 0.0;
    basis0[9][1] = a2;
    basis0[9][2] = bbb*0.5 + bb + bb;

    basis0[10][0] = a2;
    basis0[10][1] = a2;
    basis0[10][2] = bbb + bb + bb;

    basis0[11][0] = a2;
    basis0[11][1] = a;
    basis0[11][2] = bbb*1.5 + bb + bb;

    double **basis;
    double E00, E, c;
    int nds = 50;
    double ds;
    FILE *out;

    basis = create(basis,nb,3,"SFp100:create()");
    E00 = crystal_eng(trans0,basis0,nb);

    c = lc_a0*5.0/4.0;
    ds = lc_a0/sqrt(2.0)/nds;

    // Full scan of (100) surface.
    out = fopen("SFp100.eng","w");
    for (int i=0; i<=nds; i++) {
      for (int j=0; j<=nds; j++) {
	for (int k=0; k<nb; k++) {
	  if (basis0[k][2]>c-0.01) {
	    basis[k][0] = basis0[k][0] + i*ds;
	    basis[k][1] = basis0[k][1] + j*ds;
	    basis[k][2] = basis0[k][2];
	  } else {
	    basis[k][0] = basis0[k][0];
	    basis[k][1] = basis0[k][1];
	    basis[k][2] = basis0[k][2];
	  }
	}
	WriteConfigs("SFp100","SFp100.dat",trans0,basis0,nb);
	//WriteDump("SFp100.dump",trans0,basis,nb,i*ds+j);
	E = crystal_eng(trans0,basis,nb);
	fprintf(out ," %f %f %f\n",i*ds,j*ds,(E-E00)*16.0218/Area);
      }
    }
    fclose(out);
    destroy(basis);
  }

  if (!ifound) {
    char buf[256];
    sprintf(buf," cannot find structure: %s",basic_struc);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }
  max1 = max2 = max3 = 1;
}


void SFp110()
{
  double Area;
  int nb, ifound = 0;

  max1 = 4;
  max2 = 4;
  max3 = 0;

  if (strcmp(basic_struc,"fcc") == 0) {
    ifound = 1;
    double a,a2,bb,bbb;
    a = lc_a0/sqrt(2.0); //Re;
    a2 = a/2.0;
    bb = a*sqrt(2.0);
    bbb = bb/2.0;
    nb = 8;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf110:create()");
    else basis0 = grow(basis0,nb,3,"surf110:grow()");

    Area = a*bb;

    trans0[0][0] = bb;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = 3.0*a;

    basis0[0][0] = 0.0;
    basis0[0][1] = 0.0;
    basis0[0][2] = 0.0;

    basis0[1][0] = bbb;
    basis0[1][1] = a2;
    basis0[1][2] = a2;

    basis0[2][0] = 0.0;
    basis0[2][1] = 0.0;
    basis0[2][2] = a;

    basis0[3][0] = bbb;
    basis0[3][1] = a2;
    basis0[3][2] = a2 + a;

    basis0[4][0] = 0.0;
    basis0[4][1] = 0.0;
    basis0[4][2] = 2*a;

    basis0[5][0] = bbb;
    basis0[5][1] = a2;
    basis0[5][2] = a2 + 2*a;

    basis0[6][0] = 0.0;
    basis0[6][1] = 0.0;
    basis0[6][2] = 3*a;

    basis0[7][0] = bbb;
    basis0[7][1] = a2;
    basis0[7][2] = a2 + 3*a;
  }

  if (strcmp(basic_struc,"bcc") == 0) {
    ifound = 1;
    double a,a2,bb,bbb;
    a = lc_a0; // 2.0*Re/sqrt(3.0);
    a2 = a/2.0;
    bb = a*sqrt(2.0);
    bbb = bb/2.0;
    nb = 10;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf110:create()");
    else basis0 = grow(basis0,nb,3,"surf110:grow()");
    Area = a*bb;
    trans0[0][0] = bb;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = 5.0*bbb;

    basis0[0][0] = 0.0;
    basis0[0][1] = 0.0;
    basis0[0][2] = 0.0;

    basis0[1][0] = bbb;
    basis0[1][1] = a2;
    basis0[1][2] = 0.0;

    basis0[2][0] = bbb;
    basis0[2][1] = 0.0;
    basis0[2][2] = bbb;

    basis0[3][0] = bb;
    basis0[3][1] = a2;
    basis0[3][2] = bbb;
    // ------ glide plane -------
    basis0[4][0] = bb;
    basis0[4][1] = 0.0;
    basis0[4][2] = bb;

    basis0[5][0] = bbb + bb;
    basis0[5][1] = a2;
    basis0[5][2] = bb;

    basis0[6][0] = bb + bbb;
    basis0[6][1] = 0.0;
    basis0[6][2] = bb + bbb;

    basis0[7][0] = 2.0*bb;
    basis0[7][1] = a2;
    basis0[7][2] = bb + bbb;

    basis0[8][0] = 2.0*bb;
    basis0[8][1] = 0.0;
    basis0[8][2] = 2.0*bb;

    basis0[9][0] = 2.0*bb + bbb;
    basis0[9][1] = a2;
    basis0[9][2] = 2.0*bb;
  }

  if (strcmp(basic_struc,"diam")==0) {
    ifound = 1;
    double a,a2,bb,bbb;
    a = lc_a0/sqrt(2.0); // 4.0*Re/sqrt(6.0);
    a2 = a/2.0;
    bb = a*sqrt(2.0);
    bbb = bb/2.0;
    nb = 20;
    if (basis0 == NULL) basis0 = create(basis0,nb,3,"surf110:create()");
    else basis0 = grow(basis0,nb,3,"surf110:grow()");

    Area = a*bb;

    trans0[0][0] = bb;
    trans0[0][1] = 0.0;
    trans0[0][2] = 0.0;

    trans0[1][0] = 0.0;
    trans0[1][1] = a;
    trans0[1][2] = 0.0;

    trans0[2][0] = 0.0;
    trans0[2][1] = 0.0;
    trans0[2][2] = 6.0*a;

    //basis0[0][0] = 0.0;
    //basis0[0][1] = 0.0;
    //basis0[0][2] = 0.0;

    basis0[0][0] = bbb*0.5;
    basis0[0][1] = 0.0;
    basis0[0][2] = a2;

    basis0[1][0] = bbb;
    basis0[1][1] = a2;
    basis0[1][2] = a2;

    basis0[2][0] = bbb*1.5;
    basis0[2][1] = a2;
    basis0[2][2] = a;

    basis0[3][0] = 0.0;
    basis0[3][1] = 0.0;
    basis0[3][2] = a;

    basis0[4][0] = bbb*1.5;
    basis0[4][1] = a2;
    basis0[4][2] = a + a;

    basis0[5][0] = bbb;
    basis0[5][1] = a2;
    basis0[5][2] = a2 + a;

    basis0[6][0] = bbb*0.5;
    basis0[6][1] = 0.0;
    basis0[6][2] = a2 + a;

    basis0[7][0] = 0.0;
    basis0[7][1] = 0.0;
    basis0[7][2] = a + a;

    basis0[8][0] = bbb;
    basis0[8][1] = a2;
    basis0[8][2] = a2 + a + a;

    basis0[9][0] = bbb*0.5;
    basis0[9][1] = 0.0;
    basis0[9][2] = a2 + a + a;
    // ------- glide plane ------
    basis0[10][0] = bbb*1.5;
    basis0[10][1] = a2;
    basis0[10][2] = a + a + a;

    basis0[11][0] = 0.0;
    basis0[11][1] = 0.0;
    basis0[11][2] = a + a + a;

    basis0[12][0] = bbb;
    basis0[12][1] = a2;
    basis0[12][2] = a2 + a + a + a;

    basis0[13][0] = bbb*0.5;
    basis0[13][1] = 0.0;
    basis0[13][2] = a2 + a + a + a;

    basis0[14][0] = bbb*1.5;
    basis0[14][1] = a2;
    basis0[14][2] = a + a + a + a;

    basis0[15][0] = 0.0;
    basis0[15][1] = 0.0;
    basis0[15][2] = a + a + a + a;

    basis0[16][0] = bbb;
    basis0[16][1] = a2;
    basis0[16][2] = a2 + a + a + a + a;

    basis0[17][0] = bbb*0.5;
    basis0[17][1] = 0.0;
    basis0[17][2] = a2 + a + a + a + a;

    basis0[18][0] = bbb*1.5;
    basis0[18][1] = a2;
    basis0[18][2] = 5.0*a;

    basis0[19][0] = 0.0;
    basis0[19][1] = 0.0;
    basis0[19][2] = 5.0*a;

    double **basis;
    double E00, E, c;
    int nds = 50;
    double dsx, dsy;
    FILE *out;

    basis = create(basis,nb,3,"SFp110:create()");
    E00 = crystal_eng(trans0,basis0,nb);

    c = lc_a0*3.0/sqrt(2.0);
    out = fopen("SFp110.eng","w");
    dsx = lc_a0/nds;
    dsy = lc_a0/sqrt(2.0)/nds;
    for (int i=0; i<=nds; i++) {
      for (int j=0; j<=nds; j++) {
	for (int k=0; k<nb; k++) {
	  if (basis0[k][2]>c-0.01) {
	    basis[k][0] = basis0[k][0] + i*dsx;
	    basis[k][1] = basis0[k][1] + j*dsy;
	    basis[k][2] = basis0[k][2];
	  } else {
	    basis[k][0] = basis0[k][0];
	    basis[k][1] = basis0[k][1];
	    basis[k][2] = basis0[k][2];
	  }
	}
	WriteConfigs("SFp110","SFp110.dat",trans0,basis0,nb);
	//WriteDump("SFp110.dump",trans0,basis,nb,i*nds+j);
	E = crystal_eng(trans0,basis,nb);
	fprintf(out ," %f %f %f\n",i*dsx,j*dsy,(E-E00)*16.0218/Area);
      }
    }
    fclose(out);

    destroy(basis);
  }

  if (!ifound) {
    char buf[256];
    sprintf(buf," cannot find structure: %s",basic_struc);
    errmsg(buf,FERR);
    if (nprocs > 1) MPI_Abort(world,errcode);
    exit(EXIT_FAILURE);
  }
  max1 = max2 = max3 = 1;
}
