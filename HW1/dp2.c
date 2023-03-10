#include <stdio.h>	/* for printf */
#include <stdint.h>	/* for uint64 definition */
#include <stdlib.h>	/* for exit() definition */
#include <time.h>	/* for clock_gettime */
#define BILLION 1000000000L


float dpunroll(long N, float *pA, float *pB) { 
    float R = 0.0;
    int j;
    for (j=0;j<N;j+=4)
        R += pA[j]*pB[j] + pA[j+1]*pB[j+1] \
           + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
    return R; 
}

int main(int argc,char** argv){
    struct timespec start, end;
    long N=atol(argv[1]);
    long repetition=atol(argv[2]);
    float * pA=(float*) malloc(N*sizeof(float));
    for(int i=0;i<N;i++){
        pA[i]=1.0;
    }

    float * pB=(float*) malloc(N*sizeof(float));
    for(int i=0;i<N;i++){
        pB[i]=1.0;
    }

    float R=0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i=0;i<repetition/2;i++){
        R=dpunroll(N, pA, pB);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("%f\n",R);
    double time_diff1 = ((double)end.tv_sec - (double)start.tv_sec) + ((double)end.tv_nsec - (double)start.tv_nsec)/BILLION;
	printf("%f\n",time_diff1);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for(int i=0;i<repetition/2;i++){
        R=dpunroll(N, pA, pB);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("%f\n",R);

    double time_diff = ((double)end.tv_sec - (double)start.tv_sec) + ((double)end.tv_nsec - (double)start.tv_nsec)/BILLION;
	printf("%f\n",time_diff);
    //printf("%.03lf,%.03lf\n", (double)end.tv_sec,(double)start.tv_sec);
    //printf("%.03lf,%.03lf\n", (double)end.tv_nsec,(double)start.tv_nsec);
    double average_time=time_diff*2/repetition;
    printf("N:%ld,<T>:%.06lf seconds,B:%.03lf GB/sec,F:%.03lf FLOP/sec",N, average_time,8*N/1e9f/average_time,N/average_time);
    free(pA);
    free(pB);
}