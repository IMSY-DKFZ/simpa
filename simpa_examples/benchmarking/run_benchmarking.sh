#!/bin/bash

help() {
echo "Usage: calculate benchmarking for [options]"
echo "For contributing, please use default"
echo "Options:"
echo "  -i, --init           First spacing to benchmark: default = 0.2mm"
echo "  -c, --cease          Final spacing to benchmark: default = 0.4mm"
echo "  -s, --step           Step between spacings: default = 0.1mm"
echo "  -f, --file           Where to store the output files: default - TODO specify where!"
echo "  -t, --time           Profile times taken: if no profile all are set"
echo "  -g, --gpu            Profile GPU usage: if no profile all are set"
echo "  -m, --memory         Profile memory usage: if no profile all are set"
echo "  -b, --table          Create pretty table with the primary results"
echo "  -h, --help           Display this help message"
exit 0
}

start=0.2
stop=0.4
step=0.1
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
 -h | --help) help
   ;;
  *) echo "Option $1 not recognized"
   ;;
 -b | --table) write_table='True'
   ;;
esac
shift 1
done

if [ ${#profiles[@]} -eq 0 ]; then
    echo "WARNING: using all three profilers by default"
    profiles=($"TIME")
    profiles+=($"GPU_MEMORY")
    profiles+=($"MEMORY")
fi

for spacing in $(seq $start $step $stop)
do
    for profile in "${profiles[@]}"
    do
        python3 performance_check.py --spacing $spacing --profile $profile --savefolder $filename
    done
done

if [ -$write_table == 'True' ]; then
    python3 create_benchmarking_table.py
fi
