#!/bin/bash

if [ "${model_name}" == "deepspeech" ];then
    ./launch_benchmark_deepspeech.sh $@
elif [ "${model_name}" == "NCF" ];then
    ./launch_benchmark_ncf.sh $@
fi
