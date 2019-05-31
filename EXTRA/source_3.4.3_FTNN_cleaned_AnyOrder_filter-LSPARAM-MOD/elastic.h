#ifndef ELASTIC_H
#define ELASTIC_H

// Deformation step for calculations of elastic constants
#define epsilon 0.005
#define gps 160.217733

double BulkModulus();
double c11();
double c33();
double c44();
double c66();
double ShearModulus(); // C11-C12
double c11pc12(); // C11+C12
double c13pc11pc33(); // C13+(C11+C33)/2
double c14pc11pc44(); // C14+(C11+C44)/2
void CopyStruct(double t0[][3],
		double **b0,
		double t[][3],
		double **b, const int n);
void TransformVector(double transold[][3],
                     double s[][3],
                     double transnew[][3]);
void TransformBasis(double **oldbasis,
		    double s[][3],
		    double **newbasis, const int n);
void InitStrain(double s[][3]);

#endif // ELASTIC_H
