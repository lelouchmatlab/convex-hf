#!/bin/bash


gpu=`qstat -n $PBS_JOBID|awk ' END { split ($NF, a, "/")2 printf ("gpu%s\n", a[2]) } '`
export THEANO_FLAGS="cuda.root=/usr/local/cuda,device=$gpu,floatX=float32,config.nvcc.fastmath=True,allow_gc=False"

python run_mnist.py
