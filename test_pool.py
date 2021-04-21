from multiprocessing import Pool
import os 

def multi_run_wrapper(args):
   return add(*args)

def add(x,y):
    sum = x + y
    print("sum", sum)
    # data.append(sum)
    return {str(x): y}

def initializer():
    # global data
    # data = {}
    return

if __name__ == "__main__":
    final_dict = {}

    pool = Pool(4, initializer, ())
    results = pool.map(multi_run_wrapper,[(1,2),(2,3),(3,4)])
    for test_dict in results:
        final_dict.update(test_dict)
    print(final_dict)