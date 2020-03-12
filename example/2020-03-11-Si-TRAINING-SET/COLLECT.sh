name='2020-03-11-TC.dat'; > $name

for i in  ISOLATED.dat  NO_DFT.dat ROSE.dat 2019-12-11-PY-ALT-ISO-ANISO-POSCAR-E-TRAIN.dat DC-AIMD-NPT.dat  DC-AIMD-NVT.dat
do
	awk '{print $0}' $i >> $name
done
