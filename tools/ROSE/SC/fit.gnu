Rc=4.5; Tc=1.0
ao=2.5; Eo=-5; a=0.1
E(x)=Eo*(1.0+a*(x/ao-1.0))*exp(-a*(x/ao-1.0))*(x-Rc)**4.0/(Tc**4+(x-Rc)**4)-0.795023
fit E(x) 'rn-e-2.dat' u 1:2 via Eo,a,ao
set xrange [2:Rc]
plot 'rn-e-1.dat', 'rn-e-2.dat',E(x)
pause 3
set print "FITTING-PARAM.dat"
print  Eo,a,ao
