===================== version 3.4.3 ========================================
Updated order of local BO parameters:
  A', alpha, B', beta, h, sigma, a, lambda.

===================== version 3.4.2 ========================================
Gi calculations:
1) Any number of Legendre polynomial orders and the orders can be 
   specified in the potential file. The orders must be in ascending order.
2) Options for Gi calculation (first integer in the first 
   line of the potential file):
   0: regular Gis; cutoff = Rc.
   1: Gis = log(regular Gis); cutoff = Rc.
   3: cutoff = 1.5 * Rc; no log() of Gis
   4: cutoff = 1.5 * Rc with log() of Gis
3) Regular Gis may be shifted by REF_GI (second double in the first 
   line of the potential file).
4) All processes compute Gis of their own atoms.

BOP parameters:
1) Parameters 'a' and 'lambda' are squared and they are put 
   at the end of the array.

Options for logistic function (third integer in the first 
line of the potential file):
1) 0 for sigmoid
2) 1 for 1/2tanh(x/2)

Options for derivatives (see command file):
1) 0 for finite difference
2) 1 for analytical

Notes:
1) For PINN, cutoff distance, rc, and range, d, are held constant.
2) Local BO parameters are strictly ordered:
   A', alpha, B', beta, a, h, lambda, sigma.
3) Simultaneous inclusion of test data has been disabled.
4) Examples of command and potential files can be found in the directory 
   'example_files'.



Command file format:
===================
line 1) <training data> <test data> - files
line 2) <input> <output> - potential files
line 3) <ftol> <gtol> <Tini> <Tend> <nstage> <niter> 
        - Tini and Tend are only relevant to simulated annealing method.
line 4) <potential type> - (0:BOP  1:NNET  2:NNET+BOP)
line 5) <offset to DFT energy>
line 6) <basic structure> <a> <b> <c> - equil. structure and its lattice constants.
line 7) <penalty for BOP> <penalty for NN> <penalty type 1 for PINN> <penalty type 2 for PINN> 
        - regularization factors
line 8) <flag> <file> - if flag==0 compute Gis else if flag==1 read Gis from 'file'.
line 9) <flag> - if zero use finite differenc else if 1 use analytical form for derivatives.
line 10) <+int> - random seed; if zero use current time else use '+int'.
line 11) <flag> - optimization method (0:DFP, 1:GA)
if GA open two more lines:
line 12) <+int> <+int> - flag for methods of crossing parents and interval 
                         to write statistics of the generation;
			 if flag==0, swap parameters;
			 if flag==1, apply weighted average.
line 13) <+int> <+int> <+int> <+int> - number of population, fittest population, 
                                       generation and mutation stages.
open number of lines equal to number of mutation stages specified in the previous line; 
must be greater than zero:
line 14) <double> <+int> - mutation size and steps.
......
...... and so forth.

Potential file format:
=====================
a) Straight BOP:
line 1) <string> <double> <double> - parameter name, value and step size
.......
.......
line 10) <string> <double> <double>
parameter names are: A, alpha, B, beta, a, h, lambda, sigma, hc and rc.

b) ANN and PINN:
line 1) <+int> <double> <+int> - Gi method, reference Gi, and logistic function  
                                 type (see above).
line 2) <+int> - number of chemical species.
open below number of lines equal to number of chemical species.
line 3) <string> <double> - element symbol and atomic mass.
......
......
line 2+#of chem sort+1) <+int> <double> <double> <double> <double> 
                        - flag, max. range, Rc, Hc, and width of Gaussians;
			  if flag==0, weights and biases from couple of lines below.
line 2+#of chem sort+2) <+int> <double> <double> .... <double> 
                        - number of orders of Legendre polynomials 
                          and the orders in ascending order.
line 2+#of chem sort+3) <+int> <double> <double> .... <double> - number of r0s and values.
line 2+#of chem sort+4) <+int> <A> <alpha> <B> <beta> <a> <h> <lambda> <sigma> 
                        - flag and estimates of local BO parameters;
			  if flag==0, do not use the parameters;
			  if flag==1, use the parameters.
line 2+#of chem sort+5) <+int> <+int> <+int> .... <+int> 
                        - number of layers, nodes in input layer, 
      			  first hidden layer and so on.

