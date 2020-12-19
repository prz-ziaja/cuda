from time import time
def timeit(f):
    def wrapper(*args,**kwargs):
        start = time()
        output = f(*args,**kwargs)
        stop = time()
        print(stop-start)
        return output
    
    return wrapper