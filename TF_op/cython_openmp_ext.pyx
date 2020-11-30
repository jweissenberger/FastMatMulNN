from cython.parallel cimport parallel
cimport openmp
import os


def controlOMP(numthread):
    cdef int num
    #openmp.omp_set_dynamic(1)
    openmp.omp_set_num_threads(numthread)
    #return openmp.omp_get_num_threads()
    with nogil, parallel():
        
        #return openmp.omp_get_num_threads()
        num = openmp.omp_get_num_threads()
        with gil:
            #os.system('echo $OMP_NUM_THREADS')
            #print(f'OpenMP_num_threads= {num}')
            return num

def getnum():
    cdef int num2
    with nogil, parallel():
        
        num2 = openmp.omp_get_num_threads()
        with gil:
            return num2