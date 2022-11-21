import time
def timeit(func, *args, return_time=False, **kwargs):
    start = time.time()
    ret = func(*args, **kwargs)
    if not return_time:
        print( (time.time()-start))
        return ret
    return time.time()-start, ret
def 時間(func, *args, return_time=False, **kwargs):
    start = time.time()
    ret = func(*args, **kwargs)
    if not return_time:
        print( (time.time()-start))
        return ret
    return time.time()-start, ret