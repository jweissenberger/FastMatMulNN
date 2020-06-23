#!/bin/bash
# run via "source build_script.sh" to access TF env vars

# path to fast matmul repo
FMMROOT=/home/ballard/gballard-fast-matmul

# linking to parallel MKL
MKL_FLAGS="-I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl"

# fast matmul parallel strategy
PAR_FLAGS="-fopenmp -D_PARALLEL_=1" # for DFS strategy (always call parallel MKL)
#PAR_FLAGS="-fopenmp -D_PARALLEL_=2" # for BFS strategy (always call sequential MKL, currently gives incorrect answer)
#PAR_FLAGS="-fopenmp -D_PARALLEL_=3" # for hybrid strategy (currently seg faults)

# to save time, these env vars should be set by each user; 
#TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
#TF_LFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
g++ -std=c++11 -shared classic_mat_mul.cc -o classic_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -shared dgemm_mat_mul.cc -o dgemm_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 $MKL_FLAGS
g++ -fpermissive -std=c++11 -shared bini_mat_mul.cc -o bini_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -I${FMMROOT}/linalg -I${FMMROOT}/util -I${FMMROOT}/algorithms $MKL_FLAGS
g++ -fpermissive -std=c++11 -shared $PAR_FLAGS schonhage_mat_mul.cc -o schonhage_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -I${FMMROOT}/linalg -I${FMMROOT}/util -I${FMMROOT}/algorithms $MKL_FLAGS
