#include <fftw3.h>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>

#define REAL 0
#define IMAG 1
#define NUM_POINTS (1024*4)


void acquire_signal(double *signal, double *theta, int fs) {
    /* Generate two sine waves of different frequencies and
     * amplitudes.
     */

    int i;
    for (i = 0; i < NUM_POINTS; ++i) {
        theta[i] = (double)i / (double)fs;
        signal[i] = 1.0*sin(50.25 * 2.0 * M_PI * theta[i]) +
                    0.5*sin(80.50 * 2.0 * M_PI * theta[i]);
    }
}

int main() {
    int fs = 1024;  // Sample rate in Hz
    unsigned flags{0};
    double *theta  = new double[NUM_POINTS];
    double *signal = new double[NUM_POINTS];

    fftw_complex result[NUM_POINTS/2+1];
    fftw_plan plan = fftw_plan_dft_r2c_1d(NUM_POINTS,
                                         signal,
                                         result,
                                         flags);
    acquire_signal(signal,theta, fs);
    fftw_execute(plan);

    //save signal and result
    std::ofstream f1,f2;
    f1.open ("signal.txt");
    f2.open("result.txt");


    for (int i=0; i<NUM_POINTS; i++){
        f1 <<theta[i]<<" "<<signal[i]<<"\n";
    }
    f1.close();
    
    for (int i=0; i< (NUM_POINTS/2+1); i++){
        double yf = 2.0/(double)(NUM_POINTS)* sqrt(result[i][REAL]*result[i][REAL]+ result[i][IMAG]* result[i][IMAG]);
        f2<< (double)i*fs/(NUM_POINTS) << " "<< yf <<"\n";
    }
    f2.close();
    // fftw_free(result);
    fftw_destroy_plan(plan);
    delete[] signal,theta;

    return 0;
}