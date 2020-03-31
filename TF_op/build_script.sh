#!/bin/bash

TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
g++ -std=c++11 -shared classic_mat_mul.cc -o classic_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -shared fast_mat_mul.cc -o fast_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl