#!/bin/bash

#EXPLANATION:  
#THIS CODE GENERATES HEADER LINES FOR A NN FILE WITH GIVEN NUMBER OF FITTING PARAM AND LAYERS 
	#APPENDS NN0.DAT WITH CORRECT NUMBER OF LAYERS AND FITTING PARAM AND SAVES TO NN1.dat

#EXAMPLES:  
#	./BUILD.sh 2 500 NN 	--> STRAIGHT NN FILE WITH 2 HIDDEN LAYER AND ~500 FITTING PARAM
#	./BUILD.sh 3 800 PINN	-->     PINN NN FILE WITH 3 HIDDEN LAYER AND ~800 FITTING PARAM

#CHECK FOR COMMAND LINE ARG
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]
then
	echo "ERROR: MISSING COMMAND LINE ARG"; exit 
else
	NL=$1; NFIT=$2; POT_TYPE=$3
	echo "NLAYER		=	"$NL
	echo "NFIT		=	"$NFIT
	echo "POT_TYPE	=	"$POT_TYPE
	NF=$(awk -v PT=$POT_TYPE 'BEGIN {{if(PT=="PINN"){print 8}}{if(PT=="NN"){print 1}}}') #NODES IN FINAL LAYER

	if [ -z "$NF" ]
	then
		echo "ERROR: UNKNOWN POT_TYPE"; exit 
	fi	
fi


IN=NN0.dat
OUT=NN1.dat 
#CHECK FOR TEMPLATE NN FILE 
if [ -f $IN ] 
then 
	echo "GENERATING NN: NN1.dat"
	awk  '{print $0}' NN0.dat > $OUT
	NLG=$(awk '{if(NR==5){print $1}}' $IN)				#NUMBER OF LG POLYNOMIALS
	NRo=$(awk '{if(NR==6){print $1}}' $IN)				#NUMBER OF Ro TERMS
	NGi=$(awk -v NLG=$NLG -v NRo=$NRo   'BEGIN {print NLG*NRo}' )	#NUMBER OF Gi
	NLT=$(awk -v NL=$NL 'BEGIN {print NL+2}' )	#TOTAL NUMBER OF LAYERS

	if [ $NL -eq 1 ]
	then
		NODES=$(awk -v NFIT=$NFIT -v M=$NGi -v  NF=$NF 'BEGIN {print int((NFIT-NF)/(M+1+NF))}' )	
		echo " "$NGi $NODES $NF >> $OUT
	else

	#NODES PER HIDDEN LAYER (QUADRATIC FORMULA)
		#NOTE: M x N x N X N X F -->     NL=3  AND NFIT=(NL-1)*N**2+(M+NL-1+F+1)*N+F
	
		A=$(awk  -v NL=$NL 'BEGIN {print (NL-1)}' )
		B=$(awk  -v M=$NGi -v  F=$NF  -v NL=$NL 'BEGIN {print (M+NL+F)}' )	#NUMBER OF Gi
		C=$(awk  -v NFIT=$NFIT  -v  NF=$NF 'BEGIN {print (NF-NFIT)/1.0}' )	
		#echo $A $B $C
		NODES=$(awk -v A=$A -v B=$B -v C=$C 'BEGIN {print int((-B+(B^2-4.0*A*C)^0.5)/(2.0*A))}')
		
		str=""$NLT" "$NGi
		for i in $(seq 1 $NL)
		do
			str=$str" "$NODES 
		done
		str=$str" "$NF 
		echo $str  
		echo " "$str >> $OUT

	fi

else
	echo "ERROR: $IN FILE DOESNT EXIST"; exit
fi 


