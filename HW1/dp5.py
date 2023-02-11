import numpy as np
import sys
import time

# for a simple loop
def dp(N,A,B):
    R = 0.0
    for j in range(0,N):
       R += A[j]*B[j]
    return R


if __name__=="__main__":
    N=int(sys.argv[1])
    repetition=int(sys.argv[2])
    A = np.ones(N,dtype=np.float32) 
    B = np.ones(N,dtype=np.float32)
    starttime=time.monotonic()
    for i in range(int(repetition/2)):
        np.dot(A,B)
    endtime=time.monotonic()
    print("time diff 1:",endtime-starttime)

    starttime=time.monotonic()
    for i in range(int(repetition/2)):
        np.dot(A,B)
    endtime=time.monotonic()
    print("time diff:",endtime-starttime)
    average_time=(endtime-starttime)*2/repetition


    print("N:",N," <T>:",average_time,'sec ',' B:',4*2*N/1e9/average_time,'GB/sec',' F:',N/average_time,'FLOP/sec')
