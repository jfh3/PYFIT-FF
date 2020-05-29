#CONCATINATE DIFFERENT ERROR MEASURES INTO SINGLE FILE
awk '{print $1,$2}'	 '00-stats-train.dat'     > temp1
awk '{print $2}'	 '00-stats-validate.dat'  > temp2
awk '{print $2}'	 '00-stats-test.dat'      > temp3
paste temp1 temp2 temp3 > 00-ERR-LOG.dat
rm temp1 temp2 temp3
