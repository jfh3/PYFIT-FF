for start in $(seq 0 3)
do 
	let "end = $start + 1"
	python3 run-correlation-analysis.py cmd_log_$start test_output/ $start:$end input/EOS/EOS-E-full-lsparam.dat 0.02 0.001 --cpu -n 1 -u &
done

wait