
>ROSE.dat
for i in */*ROSE*
do
	awk '{print $0}' $i >> ROSE.dat
done
