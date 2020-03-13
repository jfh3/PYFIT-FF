
N=200
E=0
name="NO_DFT"
file1=$name'.dat'
> $file1

for i in $(seq 1 $N);
do 

r=$(awk -v i=$i -v N=$N 'BEGIN {print 1.5+i*(4.45-1.75)/N}')	#NN dist
echo $r

#-------------------------------------------------
#DC 
#-------------------------------------------------
x1=$(awk -v  x=$r 'BEGIN {print x/0.433012/2}')  #r_nn -->a/2
x2=$(awk -v  x=$x1 'BEGIN {print x/2}')  	 #a/2 --> a/4
rm TEMP.dat
cat >TEMP.dat <<!
$name
1.000000
$x1  $x1  0.0 
0.0  $x1  $x1
$x1  0.0  $x1
2
carestian or direct (scaled), only the first letter matters
0.0000000000 0.0000000000 0.0000000000
$x2 $x2 $x2
$E
!
awk '{print $0}' 'TEMP.dat' >> $file1
#-------------------------------------------------


#-------------------------------------------------
#BCC 
#-------------------------------------------------
#sqrt(3)/2= 0.866025403     rnn=0.866025*a
a=$(awk  -v  r=$r 'BEGIN {print r/0.866025403}')  #r_nn -->a
a2=$(awk -v  r=$r 'BEGIN {print r/0.866025403/2}')  #r_nn -->a
rm TEMP.dat
cat >TEMP.dat <<!
$name
1.000000
$a   0.0  0.0 
0.0  $a   0.0
0.0  0.0  $a
2
carestian or direct (scaled), only the first letter matters
0.0000000000 0.0000000000 0.0000000000
$a2 $a2 $a2
$E
!
awk '{print $0}' 'TEMP.dat' >> $file1

#-------------------------------------------------
#SC 
#-------------------------------------------------
rm TEMP.dat
cat >TEMP.dat <<!
$name
1.000000
$r   0.0  0.0 
0.0  $r   0.0
0.0  0.0  $r
1
carestian or direct (scaled), only the first letter matters
0.0000000000 0.0000000000 0.0000000000
$E
!
awk '{print $0}' 'TEMP.dat' >> $file1
#-------------------------------------------------


#-------------------------------------------------
#FCC 
#-------------------------------------------------
#sqrt(2)/2= 0.7071067811     rnn=0.7071067811*a
a=$(awk  -v  r=$r 'BEGIN {print r/0.707106}')  #r_nn -->a
a2=$(awk -v  r=$r 'BEGIN {print r/0.707106/2}')  #r_nn -->a
rm TEMP.dat
cat >TEMP.dat <<!
$name
1.000000
$a   0.0  0.0 
0.0  $a   0.0
0.0  0.0  $a
4
carestian or direct (scaled), only the first letter matters
0    0  0
$a2 $a2 0
0   $a2 $a2  
$a2  0  $a2  
$E
!
awk '{print $0}' 'TEMP.dat' >> $file1

#-------------------------------------------------


#-------------------------------------------------
#DIMER 
#-------------------------------------------------
v=$(awk  -v  r=$r 'BEGIN {print 20+r}')  #r_nn -->a
rm TEMP.dat
cat >TEMP.dat <<!
$name
1.000000
$v   0.0  0.0 
0.0  20.0 0.0
0.0  0.0  20.0
2
carestian or direct (scaled), only the first letter matters
0.0000000000 0.0000000000 0.0000000000
0.0000000000 $r		  0.0000000000
$E
!
awk '{print $0}' 'TEMP.dat' >> $file1
#-------------------------------------------------



done

rm TEMP.dat
