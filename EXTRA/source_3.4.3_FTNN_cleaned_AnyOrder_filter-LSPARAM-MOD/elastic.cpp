#include <math.h>

#include "globals.h"
#include "elastic.h"
#include "crystal_struc.h"
#include "compute.h"

double BulkModulus()
{
  //------------------
  // Bulk modulus: B
  //------------------

  double trans[3][3],strain[3][3],**basis;
  int n0;
  double E, omega;
  double q11,r1;
  double x,y,Q1;
  int nb;

  set_struc(basic_struc,1.0,nb);
  omega = VolumePerAtom(trans0,nb);
  basis = create(basis,nb,3,"BulkModulus:create()");
  n0 = (int)(nelast/2);
  InitStrain(strain);
  q11 = 0.0;
  r1 = 0.0;

  for (int ii=1; ii<=nelast; ii++) {
    x = (ii - 1 - n0)*epsilon;
    y = x*100;

    /* Construct the strain matrix */
    strain[0][0] = 1.0 + x;
    strain[1][1] = 1.0 + x;
    strain[2][2] = 1.0 + x;

    /* Transform translation vectors */
    TransformVector(trans0,strain,trans);

    /* Transform basis vectors */
    TransformBasis(basis0,strain,basis,nb);

    /* Compute the deformed energy */

    E = crystal_eng(trans,basis,nb)/nb;
    q11 = q11 + pow(y,4);
    r1 = r1 + (E - E0)*y*y;
  }

  Q1 = r1/q11;
  destroy(basis);

  return (gps*2.0*Q1/omega/9.0*10000);
}


double c11()
{
  // Elastic constant: C11
  double trans[3][3],strain[3][3],**basis;
  int n0;
  double E, omega;
  double q11,r1;
  double x,Q1;
  int nb;

  set_struc(basic_struc,1.0,nb);
  omega = VolumePerAtom(trans0,nb);
  basis = create(basis,nb,3,"c11:create()");
  n0 = (int)(nelast/2);
  //std::cout << " n0 = " << n0 << "\n";
  InitStrain(strain);
  q11 = 0.0;
  r1 = 0.0;

  for (int ii=1; ii<=nelast; ii++) {
    x = (ii - 1 - n0)*epsilon;

    // Construct the strain matrix
    strain[0][0] = 1.0 + x;

    // Transform translation vectors
    TransformVector(trans0,strain,trans);

    // Transform basis vectors
    TransformBasis(basis0,strain,basis,nb);

    // Compute the deformed energy
    E = crystal_eng(trans,basis,nb)/nb;

    q11 = q11 + pow(x,4);
    r1 = r1 + (E - E0)*x*x;
  }

  Q1 = r1/q11;
  destroy(basis);

  return (gps*2.0*Q1/omega);
}


double c33()
{
  // Elastic constant: C33

  double trans[3][3],strain[3][3],**basis;
  int n0;
  double E, omega;
  double q11,r1;
  double x,Q1;
  int nb;

  set_struc(basic_struc,1.0,nb);
  omega = VolumePerAtom(trans0,nb);

  basis = create(basis,nb,3,"Elastic:create()");
  n0 = (int)(nelast/2);
  InitStrain(strain);
  q11 = 0.0;
  r1 = 0.0;

  for (int ii=1; ii<=nelast; ii++) {
    x = (ii - 1 - n0)*epsilon;

    // Construct the strain matrix
    strain[2][2] = 1.0 + x;

    // Transform translation vectors
    TransformVector(trans0,strain,trans);

    // Transform basis vectors
    TransformBasis(basis0,strain,basis,nb);

    // Compute the deformed energy
    E = crystal_eng(trans,basis,nb)/nb;

    q11 = q11 + pow(x,4);
    r1 = r1 + (E - E0)*x*x;
  }

  Q1 = r1/q11;
  destroy(basis);

  return gps*2.0*Q1/omega;
}


double c44()
{
  // Elastic constant: C44

  double trans[3][3],strain[3][3],**basis;
  int n0;
  double E, omega;
  double q11,r1;
  double x,Q1;
  int nb;

  set_struc(basic_struc,1.0,nb);
  omega = VolumePerAtom(trans0,nb);
  basis = create(basis,nb,3,"Elastic:create()");
  n0 = (int)(nelast/2);
  InitStrain(strain);
  q11 = 0.0;
  r1 = 0.0;

  for (int ii=1; ii<=nelast; ii++) {
    x = (ii - 1 - n0)*epsilon;

    // Construct the strain matrix
    strain[1][2] = 0.5*x;
    strain[2][1] = strain[1][2];
    // strain[0][0] = 1.0 + x2/(4.0 - x2);

    // Transform translation vectors
    TransformVector(trans0,strain,trans);

    // Transform basis vectors
    TransformBasis(basis0,strain,basis,nb);

    // Compute the deformed energy
    E = crystal_eng(trans,basis,nb)/nb;

    q11 = q11 + pow(x,4);
    r1 = r1 + (E - E0)*x*x;
  }

  Q1 = r1/q11;
  destroy(basis);

  return gps*2.0*Q1/omega;
}


double c66()
{
  // Elastic constant: C66

  double trans[3][3],strain[3][3],**basis;
  int n0;
  double E, omega;
  double q11,r1;
  double x,Q1;
  int nb;

  set_struc(basic_struc,1.0,nb);
  omega = VolumePerAtom(trans0,nb);
  basis = create(basis,nb,3,"Elastic:create()");
  n0 = (int)(nelast/2);
  InitStrain(strain);
  q11 = 0.0;
  r1 = 0.0;

  for (int ii=1; ii<=nelast; ii++) {
    x = (ii - 1 - n0)*epsilon;

    // Construct the strain matrix
    strain[0][1] = 0.5*x;
    strain[1][0] = strain[0][1];

    // Transform translation vectors
    TransformVector(trans0,strain,trans);

    // Transform basis vectors
    TransformBasis(basis0,strain,basis,nb);

    // Compute the deformed energy
    E = crystal_eng(trans,basis,nb)/nb;

    q11 = q11 + pow(x,4);
    r1 = r1 + (E - E0)*x*x;
  }

  Q1 = r1/q11;
  destroy(basis);

  return gps*2.0*Q1/omega;
}


double ShearModulus()
{
  // Shear modulus: C11 - C12

  double trans[3][3],strain[3][3],**basis;
  int n0;
  double E, omega;
  double q11,r1;
  double x,Q1;
  int nb;

  set_struc(basic_struc,1.0,nb);
  omega = VolumePerAtom(trans0,nb);
  basis = create(basis,nb,3,"Elastic:create()");
  n0 = (int)(nelast/2);
  InitStrain(strain);
  q11 = 0.0;
  r1 = 0.0;

  for (int ii=1; ii<=nelast; ii++) {
    x = (ii - 1 - n0)*epsilon;

    // Construct the strain matrix
    strain[0][0] = 1.0 + x;
    strain[1][1] = 1.0 - x;
    // strain[2][2] = 1.0 + x2/(1.0 - x2);

    // Transform translation vectors
    TransformVector(trans0,strain,trans);

    // Transform basis vectors
    TransformBasis(basis0,strain,basis,nb);

    // Compute the deformed energy
    E = crystal_eng(trans,basis,nb)/nb;

    q11 = q11 + pow(x,4);
    r1 = r1 + (E - E0)*x*x;
  }

  Q1 = r1/q11;
  destroy(basis);

  return gps*Q1/omega;
}


double c11pc12()
{
  // C11 + C12

  double trans[3][3],strain[3][3],**basis;
  int n0;
  double E, omega;
  double q11,r1;
  double x,Q1;
  int nb;

  set_struc(basic_struc,1.0,nb);
  omega = VolumePerAtom(trans0,nb);
  basis = create(basis,nb,3,"Elastic:create()");
  n0 = (int)(nelast/2);
  InitStrain(strain);
  q11 = 0.0;
  r1 = 0.0;

  for (int ii=1; ii<=nelast; ii++) {
    x = (ii - 1 - n0)*epsilon;

    // Construct the strain matrix
    strain[0][0] = 1.0 + x;
    strain[1][1] = 1.0 + x;

    // Transform translation vectors
    TransformVector(trans0,strain,trans);

    // Transform basis vectors
    TransformBasis(basis0,strain,basis,nb);

    // Compute the deformed energy
    E = crystal_eng(trans,basis,nb)/nb;

    q11 = q11 + pow(x,4);
    r1 = r1 + (E - E0)*x*x;
  }

  Q1 = r1/q11;
  destroy(basis);

  return gps*Q1/omega;
}


double c13pc11pc33()
{
  // C13 + (C11 + C33)/2

  double trans[3][3],strain[3][3],**basis;
  int n0;
  double E, omega;
  double q11,r1;
  double x,Q1;
  int nb;

  set_struc(basic_struc,1.0,nb);
  omega = VolumePerAtom(trans0,nb);
  basis = create(basis,nb,3,"Elastic:create()");
  n0 = (int)(nelast/2);
  InitStrain(strain);
  q11 = 0.0;
  r1 = 0.0;

  for (int ii=1; ii<=nelast; ii++) {
    x = (ii - 1 - n0)*epsilon;

    // Construct the strain matrix
    strain[0][0] = 1.0 + x;
    strain[2][2] = 1.0 + x;

    // Transform translation vectors
    TransformVector(trans0,strain,trans);

    // Transform basis vectors
    TransformBasis(basis0,strain,basis,nb);

    // Compute the deformed energy
    E = crystal_eng(trans,basis,nb)/nb;

    q11 = q11 + pow(x,4);
    r1 = r1 + (E - E0)*x*x;
  }

  Q1 = r1/q11;
  destroy(basis);

  return gps*Q1/omega;
}

double c14pc11pc44()
{
  // C14 + (C11 + C44)/2

  double trans[3][3],strain[3][3],**basis;
  int n0;
  double E,omega;
  double q11,r1;
  double x,Q1;
  int nb;

  set_struc(basic_struc,1.0,nb);
  omega = VolumePerAtom(trans0,nb);
  basis = create(basis,nb,3,"Elastic:create()");
  n0 = (int)(nelast/2);
  InitStrain(strain);
  q11 = 0.0;
  r1 = 0.0;

  for (int ii=1; ii<=nelast; ii++) {
    x = (ii - 1 - n0)*epsilon;

    // Construct the strain matrix
    strain[0][0] = 1.0 + x;
    strain[2][1] = 0.5*x;
    strain[1][2] = strain[2][1];

    // Transform translation vectors
    TransformVector(trans0,strain,trans);

    // Transform basis vectors
    TransformBasis(basis0,strain,basis,nb);

    // Compute the deformed energy
    E = crystal_eng(trans,basis,nb)/nb;

    q11 = q11 + pow(x,4);
    r1 = r1 + (E - E0)*x*x;
  }

  Q1 = r1/q11;
  destroy(basis);

  return gps*Q1/omega;
}

void CopyStruct(double t0[][3], double **b0,
		double t[][3], double **b, const int n)
{
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) t[i][j] = t0[i][j];
    for (int j=0; j<n; j++) b[j][i] = b0[j][i];
  }
}


void TransformVector(double transold[][3],
                     double s[][3],
                     double transnew[][3])
{
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      transnew[i][j] = 0.0;
      for (int k=0; k<3; k++) transnew[i][j] = transnew[i][j] + transold[i][k]*s[j][k];
    }
  }
}


void TransformBasis(double **oldbasis,
                    double s[][3],
		    double **newbasis, const int n)
{
  for (int i=0; i<n; i++) {
    for (int j=0; j<3; j++) {
      newbasis[i][j] = 0.0;
      for (int k=0; k<3; k++) newbasis[i][j] = newbasis[i][j] + oldbasis[i][k]*s[j][k];
    }
  }
}


void InitStrain(double s[][3])
{
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      if (i!=j) s[i][j] = 0.0;
      else s[i][j] = 1.0;
    }
  }
}
