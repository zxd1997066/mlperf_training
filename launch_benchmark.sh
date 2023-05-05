#!/bin/bash
set -xe

model_name=$(echo $@ |sed 's/.*--model_name.//;s/ *--.*//')

if [ "${model_name}" == "deepspeech" ];then
    ./launch_benchmark_deepspeech.sh $@
elif [ "${model_name}" == "NCF" ];then
    ./launch_benchmark_ncf.sh $@
fi