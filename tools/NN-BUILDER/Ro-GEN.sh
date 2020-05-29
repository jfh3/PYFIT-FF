#!/bin/bash

#EXPLANATION:  
#THIS CODE SAMPLES Ro (EITHER INSIDE OR OUTSIDE

#CHECK FOR COMMAND LINE ARG
if [ -z "$1" ] || [ -z "$2" ]  || [ -z "$3" ] || [ -z "$4" ] 
then
	echo "ERROR: MISSING COMMAND LINE ARG"; exit 
else

	
	INSIDE=$1; Ro=$2 Rc=$3; Nro=$4
	echo "INSIDE			=	"$INSIDE
	echo "Ro (angs)		=	"$Ro
	echo "Rc (angs)		=	"$Rc
	echo "Nro			=	"$Nro
	dr=$(awk -v Ro=$Ro -v Rc=$Rc -v Nro=$Nro 'BEGIN {print (Rc-Ro)/Nro}')

	if [ $INSIDE -eq 1 ]
	then
		echo "BUILD RO INSIDE RANGE: $Ro-$Rc"
		#echo $dr
		out="$Nro"
		for i in $(seq $Nro)
		do
			ro=$(awk -v Ro=$Ro -v i=$i -v dr=$dr 'BEGIN {print (Ro+(i-1)*dr)/1.0}')
			out="$out $ro" 
			#echo $i
		done
		echo $out
	else
		echo "BUILD RO OUTSIDE RANGE: $Ro-$Rc" 

		out="$Nro"


		i=1

		Nro2=$(awk -v Nro=$Nro 'BEGIN {print int(1+Nro/2)}')
		while [ $i -le $Nro2 ]
		do
			ro=$(awk -v Ro=$Ro -v i=$i -v dr=$dr 'BEGIN {print (Ro-(i-1)*dr)/1.0}');	 out="$out $ro" 
			i=$(( i+1 ))	 # increments $n
		done

		while [ $i -le $Nro ]
		do
			ro=$(awk -v Rc=$Rc -v i=$i -v dr=$dr 'BEGIN {print (Rc+(i-1)*dr)/1.0}');	 out="$out $ro" 
			i=$(( i+1 ))	 # increments $n
		done
		echo $out

	fi


fi


exit

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


