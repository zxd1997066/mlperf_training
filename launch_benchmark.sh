#!/bin/bash
set -xe

if [ "${model_name}" == "deepspeech" ];then
    echo $@
    ./launch_benchmark_deepspeech.sh $@
elif [ "${model_name}" == "NCF" ];then
    ./launch_benchmark_ncf.sh $@
fi