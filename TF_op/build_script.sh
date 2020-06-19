#!/bin/bash

FMMROOT=/home/ballard/gballard-fast-matmul

TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
MKL_FLAGS="-I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
g++ -std=c++11 -shared classic_mat_mul.cc -o classic_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -shared dgemm_mat_mul.cc -o dgemm_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 $MKL_FLAGS
g++ -fpermissive -std=c++11 -shared bini_mat_mul.cc -o bini_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -I${FMMROOT}/linalg -I${FMMROOT}/util -I${FMMROOT}/algorithms $MKL_FLAGS
g++ -fpermissive -std=c++11 -shared schonhage_mat_mul.cc -o schonhage_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -I${FMMROOT}/linalg -I${FMMROOT}/util -I${FMMROOT}/algorithms $MKL_FLAGS
