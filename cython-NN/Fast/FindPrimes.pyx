import cython
cimport numpy as np
import numpy as np




def find_primes(int num) :
    cdef np.ndarray prime 
    prime = np.ones(num)
    cdef int c = 0
    cdef int p = 2
    while (p * p <= num):
        

        if (prime[p] == 1):
            c = p*p
            while(c < num):
                prime[c] = 0
                c+=p
        p += 1
    return prime