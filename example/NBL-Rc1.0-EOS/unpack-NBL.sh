grep NB Nbdlist.dat > NBL.dat
awk '{for (i=3;i<=NF;i++) {if($i<0.000001 && $i>-0.0000001 ){print 0}else{print $i/1.0}}}' NBL.dat  > TEMP
sort -n -k1 TEMP > NBL.dat 
rm TEMP
