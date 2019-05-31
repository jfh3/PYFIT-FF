#ifndef DERIVATIVES_H
#define DERIVATIVES_H

void compute_SMat();
void Init_SMat();
void iCompute_NN_Deriv(const int pottype, MatDoub &g);
void computeNNDeriv(VecDoub &x, VecDoub &g);
void computePINNDeriv(VecDoub &x, VecDoub &g);

#endif //DERIVATIVES_H
