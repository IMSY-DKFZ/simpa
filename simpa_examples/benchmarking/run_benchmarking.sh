#!/bin/bash

set -e

help() {
echo "Usage: calculate benchmarking for [options]"
echo "For further details see readme"
echo "Number of examples can be selected in performance_check.py"
echo "For comparable benchmarks, please use default"
echo "Options:"
echo "  -i, --init           First spacing to benchmark: default = 0.2mm"
echo "  -c, --cease          Final spacing to benchmark: default = 0.25mm"
echo "  -s, --step           Step between spacings: default = 0.05mm"
echo "  -f, --file           Where to store the output files: default save in current directory; 'print' prints it in console"
echo "  -t, --time           Profile times taken: if no profile, all are set"
echo "  -g, --gpu            Profile GPU usage: if no profile, all are set"
echo "  -m, --memory         Profile memory usage: if no profile, all are set"
echo "  -n, --number         Number of simulations: default = 1"
echo "  -h, --help           Display this help message"
exit 0
}

start=0
stop=0
step=0
number=1
profiles=()
filename='default'

while [ -n "$1" ]; do
case "$1" in
 -i | --init) start=$2
 shift 1
   ;;
 -c  | --cease) stop=$2
 shift 1
   ;;
 -s | --step) step=$2
 shift 1
   ;;
 -f | --file) filename=$2
 shift 1
   ;;
 -t | --time) profiles+=("TIME")
   ;;
 -g | --gpu) profiles+=("GPU_MEMORY")
   ;;
 -m | --memory) profiles+=("MEMORY")
    ;;
 -n | --number) number=$2
 shift 1
   ;;
 -h | --help) help
   ;;
  *) echo "Option $1 not recognized"
   ;;
esac
shift 1
done

if [ "$start" == 0 ]; then
  start=0.2
fi

if [ "$stop" == 0 ]; then
  stop=0.25
fi

if [ "$step" == 0 ]; then
  step=0.05
fi

if [ ${#profiles[@]} -eq 0 ]; then
    echo "WARNING: using all three profilers by default"
    profiles=($"TIME")
    profiles+=($"GPU_MEMORY")
    profiles+=($"MEMORY")
fi

prfs=''
for profile in "${profiles[@]}"
do
  prfs+="$profile"
  prfs+="%"
done

for ((i=0; i < number; i++))
do
  for spacing in $(LC_NUMERIC=C seq $start $step $stop)
  do
      for profile in "${profiles[@]}"
      do
          python3 performance_check.py --spacing $spacing --profile $profile --savefolder $filename
      done
  done
  python3 extract_benchmarking_data.py --start $start --stop $stop --step $step --profiles "$prfs" --savefolder $filename
done

python3 get_final_table.py --savefolder $filename
