import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import sys
import time
import math

def test(Dim=(7,5)):
    a = np.random.randn(Dim[0],Dim[1]).astype(np.float32)
    Dim = np.int32(Dim)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    Dim_gpu = cuda.mem_alloc(Dim.nbytes)
    cuda.memcpy_htod(Dim_gpu, Dim)

    mod = SourceModule("""
      __global__ void doublify(float *a, int *Dim) {
        int idx = threadIdx.x + threadIdx.y*Dim[0];
        a[idx] *= 2;
      }""")

    func = mod.get_function("doublify")
    import time
    s = time.time()
    func(a_gpu, Dim_gpu, block=(int(Dim[0]),int(Dim[1]),1))
    print("Time (in ms): {0:.4f}".format(1000*(time.time()-s)))
    a_doubled = np.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)
    print(np.allclose(a*2,a_doubled))
    # print(a*2)
    # print(a_doubled)

def rgb_to_grey_cpu(img_rgb):
    s = time.time()
    img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    ms_time = 1000*(time.time()-s)
    print("CPU Time (in ms): {0:.4f}".format(ms_time))
    cv2.imwrite('op_py.jpg', img_grey)
    return ms_time

def rgb_to_grey_gpu(img_rgb):
    # BGR to R, G, B
    img_b, img_g, img_r = cv2.split(img_rgb)
    img_r_gpu = cuda.mem_alloc(img_r.nbytes)
    img_g_gpu = cuda.mem_alloc(img_g.nbytes)
    img_b_gpu = cuda.mem_alloc(img_b.nbytes)
    # img_rgb_gpu = cuda.mem_alloc(img_rgb.nbytes)
    cuda.memcpy_htod(img_r_gpu, img_r)
    cuda.memcpy_htod(img_g_gpu, img_b)
    cuda.memcpy_htod(img_b_gpu, img_b)
    # # cuda.memcpy_htod(img_rgb_gpu, img_rgb)
    rows, cols, _ = img_rgb.shape
    Dim = np.int32((rows, cols))
    Dim_gpu = cuda.mem_alloc(Dim.nbytes)
    cuda.memcpy_htod(Dim_gpu, Dim)

    img_grey = np.zeros((rows, cols), dtype=np.uint8)
    img_grey_gpu = cuda.mem_alloc(img_grey.nbytes)

    mod = SourceModule("""
      __global__ void rgb_to_grey(unsigned char *img_r_gpu, unsigned char *img_g_gpu, unsigned char *img_b_gpu, unsigned char *img_grey_gpu) {
          int index_x = blockIdx.x * blockDim.x + threadIdx.x;
          int index_y = blockIdx.y * blockDim.y + threadIdx.y;
          int grid_width = gridDim.x * blockDim.x;
          int index = index_y * grid_width + index_x;
          img_grey_gpu[index] =  (int)(.299f * img_r_gpu[index] + .587f * img_g_gpu[index] + .114f * img_b_gpu[index]);
      } """)
    func = mod.get_function("rgb_to_grey")
    threads = 8
    grid_size = (math.ceil(rows/threads),math.ceil(cols/threads),1)
    import time
    s = time.time()
    func(img_r_gpu, img_g_gpu, img_b_gpu, img_grey_gpu, block=(threads,threads,1), grid=grid_size)
    ms_time = 1000*(time.time()-s)
    print("GPU Time (in ms): {0:.4f}".format(ms_time))
    cuda.memcpy_dtoh(img_grey, img_grey_gpu)
    cv2.imwrite('op_py.jpg', img_grey)
    return ms_time

if __name__ == "__main__":
    # test()
    try:
        input_path = sys.argv[1]
        img_rgb = cv2.imread(input_path)
        print(img_rgb.shape)
        t1 = rgb_to_grey_cpu(img_rgb)
        t2 = rgb_to_grey_gpu(img_rgb)
        print("Cuda Speedup: {0:.3f}".format(t1/t2))
    except:
        print("Error in image given")

# Sample output:
# (313, 557, 3)
# CPU Time (in ms): 0.4499
# GPU Time (in ms): 0.3104
# Cuda Speedup: 1.449

# python color_to_grey_gpu.py cinque_terre_small.jpg
