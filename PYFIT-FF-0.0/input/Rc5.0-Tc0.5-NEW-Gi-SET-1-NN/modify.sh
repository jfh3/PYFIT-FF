rm LSParam-mod.dat
awk '{if($1!="NBL"){print $0}}' 'Nbdlist.dat' > LSParam-mod.dat
