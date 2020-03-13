name='2020-03-13-BULK_ISO+DIMER+ROSE+ISOLATED.dat'; > $name

for i in  BULK_ISO+Cij+DIMER.dat ISOLATED.dat  NO_DFT.dat  ROSE.dat
do
	awk '{print $0}' $i >> $name
done
