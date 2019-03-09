def rmse(a,b):
    s = 0
    for i in range(len(a)):
        s+=(a[i]-b[i])**2
    return s**0.5
