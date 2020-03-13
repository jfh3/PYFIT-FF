name='2020-03-13-BULK_ISO+ROSE+ISOLATED.dat'; > $name

for i in  BULK-ISO+Cij.dat ISOLATED.dat  NO_DFT.dat  ROSE-SC-FCC-BCC-DC.dat
do
	awk '{print $0}' $i >> $name
done
