#!/bin/bash
set -xe

model_name=$(echo $@ |sed 's/.*--model_name.//;s/ *--.*//')

if [ "${model_name}" == "deepspeech" ];then
    ./launch_benchmark_deepspeech.sh $@
elif [ "${model_name}" == "NCF" ];then
    ./launch_benchmark_ncf.sh $@
elif [ "${model_name}" == "ssd300" ];then
    git clone -b ssd300 https://github.com/zxd1997066/mlperf_training.git mlperf_training
    cd mlperf_training/
    ./launch_benchmark.sh $@
fi