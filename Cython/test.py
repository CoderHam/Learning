import timeit

cy_time = timeit.timeit('RMSE_cy.rmse(a,a)', setup='import RMSE_cy; import numpy as np; a = np.arange(1e6)',number=100)
py_np_time = timeit.timeit('np.sqrt(np.mean((a-a)**2))', setup='import numpy as np; a = np.arange(1e6)',number=100)
py_slow_time = timeit.timeit('RMSE_py.rmse(a,a)', setup='import RMSE_py; import numpy as np; a = np.arange(1e6).tolist();',number=100)

print("Pure Python: {0:.2f} s\nNumpy Python: {1:.2f} s\nCython: {2:.2f} s".format(py_slow_time, py_np_time, cy_time))
print("How many times Cython is Faster than pure Python: {0:.2f}x\nHow many times Cython is Faster than Numpy Python: {1:.2f}x".format(py_slow_time/cy_time, py_np_time/cy_time))
