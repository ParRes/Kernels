if [ $# -ne 5 ]; then
  echo Usage: $0 executable \#ranks \#spare_ranks \#rank_kills \"executable parameters\" \(surrounded by string quotes\)
  exit
fi

PID=$$
EXE=$1
NUMRANKS=$2
NUMSPARES=$3
NUMKILLS=$4
PARMS=$5

echo Running FT harness for Stencil code, "#ranks=$NUMRANKS, #spares=$NUMSPARES #kills=$NUMKILLS"

#test sanity of input
if [ $NUMSPARES -lt $NUMKILLS ]; then
  echo Attempting to kill more ranks \($NUMKILLS\) than we have spares available \($NUMSPARES\)
  exit
elif [ $NUMRANKS -le $NUMSPARES ]; then
  echo Attempting to reserve too many spare ranks \($NUMSPARES\) of the total number \($NUMRANKS\)
  exit
fi

#start program and collect list of (hostname, pid) pairs in a file
rm -f __thislog.$PID
touch __thislog.$PID
 ~/ulfm-install/bin/mpirun -np $NUMRANKS $EXE $PARMS $NUMSPARES | tee __thislog.$PID &
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

#do the actual killing of ranks
localhostname=`hostname -A`
i=0
while [ $i -lt $NUMKILLS ]; do
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
  sleep 1
done

rm -f __thislog.$PID __hostpidlist.$PID
wait
