import numpy as np

def find_primes(num) :
    prime = np.ones(num)

    p = 2
    while (p * p <= num):

        if (prime[p] == 1):
            for i in range(p * p, num, p):
                prime[i] = 0
        p += 1
    return prime