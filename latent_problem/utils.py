from time import time

def time_process(func):
    def res(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        print(f"Running time of {func.__name__}: {end - start} sec")
    return res
