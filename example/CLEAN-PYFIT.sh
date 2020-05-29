loc2=${PWD}


#CHECK FOR INPUT VARIABLE


if [ -f "CLEANED.dat" ] 
then 
	echo "ALREADY CLEANED"
else 

	if [ -f "00-NN-111111.dat" ] 
	then 
		rm 00-e*0.dat
		rm log 
#		gzip 00-err-log.dat
		> CLEANED.dat
	else

		echo "DOESNT EXISTS"

		ls -ltr 00-e_vs_V-validate* > temp 
		awk '{print $9}' temp > temp2 
		file1=$(tail -n 1 temp2)
		echo $file1
		mv $file1 00-e_vs_V-validate-111111.dat
		rm temp temp2 

		ls -ltr 00-e_vs_V-train* > temp 
		awk '{print $9}' temp > temp2 
		file1=$(tail -n 1 temp2)
		echo $file1
		mv $file1 00-e_vs_V-train-111111.dat
		rm temp temp2 

		ls -ltr 00-e_vs_V-no_dft-* > temp 
		awk '{print $9}' temp > temp2 
		file1=$(tail -n 1 temp2)
		echo $file1
		mv $file1 00-e_vs_V-no_dft-111111.dat
		rm temp temp2 

		ls -ltr 00-e_vs_V-test* > temp 
		awk '{print $9}' temp > temp2 
		file1=$(tail -n 1 temp2)
		echo $file1
		mv $file1 00-e_vs_V-test-111111.dat
		rm temp temp2 

		ls -ltr 00-NN-* > temp 
		awk '{print $9}' temp > temp2 
		file1=$(tail -n 1 temp2)
		echo $file1
		mv $file1 00-NN-111111.dat
		rm temp temp2 

		rm log 
#		gzip 00-err-log.dat
		rm 00-e*0.dat
		> CLEANED.dat

	fi 
ln -s 00-NN-111111.dat  NN-111111.dat

fi
#ln -s 00-NN-111111.dat  NN-111111.dat
