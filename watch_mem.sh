while true
do
	cat /proc/23056/status | grep Peak 2>&1 | tee -a memlog.txt
	sleep 60
done
