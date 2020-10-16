CC=g++
CFLAGS=-fpermissive -std=c++11 -shared -fPIC -O3 -fopenmp

# check for MKLROOT environment variable for MKL
# to set, run "source /opt/intel/mkl/bin/mklvars.sh intel64"
ifndef MKLROOT
$(error MKLROOT is undefined)
endif

# check for TF_CFLAGS and TF_LFLAGS environment variables for TensorFlow
# to set, run "export TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')"
ifndef TF_CFLAGS
$(error TF_CFLAGS is undefined)
endif
# to set, run "export TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"
ifndef TF_LFLAGS
$(error TF_LFLAGS is undefined)
endif

# compiling with and linking to parallel MKL
MKL_FLAGS=-I$(MKLROOT)/include -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

# path to fast matmul repo
FMMROOT=/home/zhanl317/fmm-gballard
# toggle fast matmul parallel strategy
#PAR_FLAGS=-fopenmp -D_PARALLEL_=1 # for DFS strategy (always call parallel MKL)
#PAR_FLAGS=-fopenmp -D_PARALLEL_=2 # for BFS strategy (always call sequential MKL)
PAR_FLAGS=-fopenmp -D_PARALLEL_=3 # for hybrid strategy 
FMM_FLAGS=-I$(FMMROOT)/linalg -I$(FMMROOT)/util -I$(FMMROOT)/algorithms $(PAR_FLAGS)

# get names of all object files to build
CC_FILES = $(wildcard src/*.cc)
OBJS = $(patsubst src/%.cc,obj/%.so,$(CC_FILES))

all: $(OBJS) #test_mkl

obj/%.so: src/%.cc 
	$(CC) $(CFLAGS) $< -o $@ $(TF_CFLAGS) $(TF_LFLAGS) $(FMM_FLAGS) $(MKL_FLAGS)

#test_mkl: test_mkl.cpp
#	$(CC) $< -o $@ $(MKL_FLAGS) -fopenmp

clean:
	rm -f obj/*.so *~  

