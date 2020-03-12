
#NOTE: HARDCODED FOR CUBIC STRUCTURES

# SC -->  v=a^3    (r1=1)  rn=a		  (r2=1)
# BCC --> v=a^3/2  (r1=2)  rn=sqrt(3)a/2  (r2=0.866025403)
# FCC --> v=a^3/4  (r1=4)  rn=sqrt(2)a/2  (r2=0.70710678)
#  DC --> v=a^3/8  (r1=8)  rn=sqrt(3)a/4  (r2=0.43301)


#RM 
rm *ROSE* rn-e-* FITTING-PARAM.dat  fit*

#INPUT
#v-e.dat --> DFT VALUES col-1=V/N col-2=E/N 
file1='v-e.dat'
N=200		#number of structures to generate
Na=8		#number of atoms in cubic cell 
name="DC-ROSE"	#group name
r1=8.0    	#relation between V/N and lattice constant --> a^3=r1*v   
r2=0.43301      #relation between nearest neighbor dist and lattice constant --> rnn=r2*a
#rmin=2.3	#min nearest neighbor distance used for fitting  
rmax=3.0	#max nearest neighbor distance used for fitting 
Rc=4.5; 
Tc=1.0
shift=0.795023

#CONVERT FILE TO NEAREST NEIGHBOR DIST
awk -v r1=$r1 -v r2=$r2 '{print r2*(r1*$1)^0.33333,$2}' $file1              > 'rn-e-1.dat'  #FULL
awk -v rmax=$rmax '{if(0.95*rmax<$1 && $1<rmax){print $1,$2}}' 'rn-e-1.dat'  > 'rn-e-2.dat'  #FITTING DATA



#WRITE GNUPLOT FITTING SCRIPT
cat > fit.gnu <<!
Rc=$Rc; Tc=$Tc
ao=2.5; Eo=-5; a=0.1
E(x)=Eo*(1.0+a*(x/ao-1.0))*exp(-a*(x/ao-1.0))*(x-Rc)**4.0/(Tc**4+(x-Rc)**4)-$shift
fit E(x) 'rn-e-2.dat' u 1:2 via Eo,a,ao
set xrange [2:Rc]
plot 'rn-e-1.dat', 'rn-e-2.dat',E(x)
pause 3
set print "FITTING-PARAM.dat"
print  Eo,a,ao
!

#FIT
gnuplot -q fit.gnu
Eo=$(awk '{print $1}' 'FITTING-PARAM.dat')
a1=$(awk '{print $2}' 'FITTING-PARAM.dat')
ao=$(awk '{print $3}' 'FITTING-PARAM.dat')
echo $Eo $a1 $ao

#WRITE EXTRPOLATED DATA SET
file1=$name'.dat';   > $file1
> fitted-data.dat
for i in $(seq 1 $N);
do 

r=$(awk -v i=$i -v rmax=$rmax -v N=$N 'BEGIN {print 1.01*rmax+i*(4.45-1.01*rmax)/N}')	#NN dist
e=$(awk -v r=$r -v Eo=$Eo -v a=$a1 -v ao=$ao -v Rc=$Rc -v Tc=$Tc -v shift=$shift 'BEGIN {print Eo*(1.0+a*(r/ao-1.0))*exp(-a*(r/ao-1.0))*(r-Rc)^4.0/(Tc^4+(r-Rc)^4)-shift}')
E=$(awk -v e=$e -v Na=$Na 'BEGIN {print e*Na}')	 
a=$(awk -v r=$r -v r2=$r2 'BEGIN {print r/r2}')	 
a2=$(awk -v a=$a 'BEGIN {print a/2}')
a4=$(awk -v a=$a 'BEGIN  {print 0.25*a}')	 
a34=$(awk -v a=$a 'BEGIN {print 0.75*a}')	 
	 


#-------------------------------------------------
#SC 
#-------------------------------------------------
emax=$(awk -v shift=$shift 'BEGIN {print -shift-0.5}')	 
if (( $(echo "$e < $emax" |bc -l) )); 
then

echo $r $e $a $E >> fitted-data.dat

cat >TEMP.dat <<!
$name
1.000000
$a   0.0  0.0 
0.0  $a   0.0
0.0  0.0  $a
$Na
carestian or direct (scaled), only the first letter matters
0.0  0.0  0.0 
$a2  $a2  0.0
$a2  0.0  $a2 
0.0  $a2  $a2
$a4  $a4  $a4
$a34 $a34 $a4
$a34 $a4  $a34 
$a4  $a34 $a34
$E
!
awk '{print $0}' 'TEMP.dat' >> $file1
#-------------------------------------------------
#exit
rm TEMP.dat
fi

done

