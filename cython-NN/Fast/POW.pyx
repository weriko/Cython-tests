
import cython
cimport numpy as np
import numpy as np
import hashlib as hs

@cython.cdivision(False)
@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def POW(bytes data, target):
    cdef int header = 1
    cdef int nonce = 0
    cdef bytes t
    while True:
        while nonce<0x7fffffff:
            t = (data+nonce.to_bytes(32,"little")+header.to_bytes(64,"little"))
            if int(hs.sha256(t).hexdigest(), 16) < target:
                return (header,nonce)
            nonce+=1
        header+=1