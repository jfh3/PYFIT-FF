name='2020-03-13-ISO-ROSE-ISOLATED.dat'; > $name

for i in  2019-11-29-PY-ISO-ONLY-POSCAR-E-TRAIN.dat  ISOLATED.dat  NO_DFT.dat  ROSE.dat
do
	awk '{print $0}' $i >> $name
done
