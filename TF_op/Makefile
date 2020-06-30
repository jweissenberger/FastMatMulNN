CC=g++
CFLAGS=-fpermissive -std=c++11 -shared -fPIC -O3

# tensorflow flags
TF_CFLAGS=-I/home/ballard/miniconda3/lib/python3.7/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0
TF_LFLAGS=-L/home/ballard/miniconda3/lib/python3.7/site-packages/tensorflow -l:libtensorflow_framework.so.2

# linking to parallel MKL (MKLROOT must be set)
MKL_FLAGS=-I$(MKLROOT)/include -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

# path to fast matmul repo
FMMROOT=/home/ballard/gballard-fast-matmul
# toggle fast matmul parallel strategy
PAR_FLAGS=-fopenmp -D_PARALLEL_=1 # for DFS strategy (always call parallel MKL)
#PAR_FLAGS=-fopenmp -D_PARALLEL_=2 # for BFS strategy (always call sequential MKL, currently gives incorrect answer)
#PAR_FLAGS=-fopenmp -D_PARALLEL_=3 # for hybrid strategy (currently seg faults)
FMM_FLAGS=-I$(FMMROOT)/linalg -I$(FMMROOT)/util -I$(FMMROOT)/algorithms $(PAR_FLAGS)

# get names of all object files to build
CC_FILES = $(wildcard src/*.cc)
OBJS = $(patsubst src/%.cc,obj/%.so,$(CC_FILES))

obj/%.so: src/%.cc 
	$(CC) $(CFLAGS) $< -o $@ $(TF_CFLAGS) $(TF_LFLAGS) $(FMM_FLAGS) $(MKL_FLAGS)

all: $(OBJS)

clean:
	rm -f obj/*.so *~  

