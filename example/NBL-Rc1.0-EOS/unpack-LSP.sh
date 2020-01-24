grep Gi Nbdlist.dat > LSP.dat
awk '{for (i=2;i<=NF;i++) {printf "%15.8f\n", $i/1.0}}' LSP.dat  > TEMP
sort -n -k1 TEMP > LSP.dat 
rm TEMP
