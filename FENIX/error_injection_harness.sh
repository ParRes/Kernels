if [ $# -ne 5 ]; then
  echo Usage: $0 \#total_ranks  \#rank_kills MTBF executable \"executable parameters\" \(surrounded by string quotes\)
  exit
fi

if [ x$ULFM_PATH == x ]; then
  echo Set environment variable ULFM_PATH to where Fault Tolerant MPI is installed
  exit
fi

if [ ! -x ./generate_times ]; then
  echo ERROR: Files \"generate times\" does not exist; please type \"make\"
  exit
fi

PID=$$
NUMRANKS=$1
NUMKILLS=$2
MTBF=$3
EXE=$4
PARMS=$5

#test sanity of input
if [ $NUMRANKS -le $NUMKILLS ]; then
  echo ERROR: Attempting to kill $NUMKILLS ranks out of $NUMRANKS
  exit
fi

if [ ! -x "$EXE" ]; then
  echo ERROR: Executable $EXE not found
  exit
fi

#issue warning about validity of timings
echo "WARNING: Make sure #spare_ranks is at least as large as #rank_kills AND"
echo "         #rank_kills is SMALLER than 50% of #total_ranks, or timings may be invalid"

echo Running error injection harness for $EXE code, "#ranks=$NUMRANKS #kills=$NUMKILLS MTBF=${MTBF}s"

#start program and collect list of (hostname, pid) pairs in a file
rm -f __thislog.$PID
touch __thislog.$PID
$ULFM_PATH/bin/mpirun -np $NUMRANKS $EXE $PARMS | tee __thislog.$PID &
# ~/ulfm-install/bin/mpirun -np $NUMRANKS $EXE $PARMS | tee __thislog.$PID &

rm -f __hostpidlist.$PID 
touch __hostpidlist.$PID
listlength=`wc -l __hostpidlist.$PID | awk '{ print $1 }'`
while [ $listlength -lt $NUMRANKS ]; do
  listlength=`wc -l __hostpidlist.$PID | awk '{ print $1 }'`
  cat  __thislog.$PID | grep __HOSTNAME__ | sort -u | awk '{ print $2" "$4 }' > __hostpidlist.$PID
done

#create separate arrays of hostnames and pids
declare -a host
declare -a pid
i=1
while [ $i -le $NUMRANKS ]; do
  host[`expr $i - 1`]=`head -$i __hostpidlist.$PID | tail -1 | awk '{ print $1 }'`
  pid[`expr $i - 1`]=`head -$i __hostpidlist.$PID | tail -1 | awk '{ print $2 }'`
  i=`expr $i + 1`
done

#wait for the root rank of the running program to give the STARTKILL signal
STARTKILL=0
while [ $STARTKILL -eq 0 ]; do
  STARTKILL=`cat __thislog.$PID | grep __FINISHED_FENIX_INIT__ | wc -l | awk '{ print $1 }'`
#  STARTKILL=`cat __thislog.$PID | grep __STARTED_ITERATIONS__ | wc -l | awk '{ print $1 }'`
done

#do the actual killing of ranks; first create the array of time intervals
declare -a nextkill
./generate_times $NUMKILLS $MTBF > __kill_times.$PID
i=0
while read -r line; do 
  nextkill[$i]=`echo $line`
  i=`expr $i + 1`
done < __kill_times.$PID

localhostname=`hostname -A`
i=0
while [ $i -lt $NUMKILLS ]; do
  sleep ${nextkill[$i]}
#  mpi_proc=`expr $NUMRANKS - $i - 1`
  mpi_proc=$i
  if [ $localhostname == ${host[$mpi_proc]} ]; then
    echo  killing local process ${pid[$mpi_proc]}
    kill -9 ${pid[$mpi_proc]}
  else
    echo killing remote process ${pid[$mpi_proc]} on host ${host[$mpi_proc]}
    ssh ${host[$mpi_proc]} kill -9 ${pid[$mpi_proc]}
  fi
  i=`expr $i + 1`
done

rm -f __thislog.$PID __hostpidlist.$PID __kill_times.$PID
wait
echo w00t, all done
