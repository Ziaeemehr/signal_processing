#include <sstream>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <assert.h>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <fftw3.h>
#include <cstdio>
#include <assert.h>
#include <cmath>

#define REAL 0
#define IMAG 1

using dim1 = std::vector<double>;
using dim2 = std::vector<std::vector<double>>;

void make_signal(dim1 &signal, const int fs);
void fft(const dim1 &signal, dim1 &freq, dim1 &freqAmplitudes, const double dt);

int main()
{
    const int fs = 1024; // Sample rate in Hz
    const double dt = 1.0 / fs;
    constexpr int NUM_POINTS = 1024;
    unsigned flags{0};

    dim1 signal(NUM_POINTS);
    dim1 freq(NUM_POINTS);
    dim1 freqAmplitudes(NUM_POINTS);

    make_signal(signal, fs);
    fft(signal, freq, freqAmplitudes, dt);



    FILE *SIGNAL_FILE = fopen("signal.txt", "w");
    FILE *FREQUENCIS = fopen("RESULT.txt", "w");
    
    for (int i = 0; i < (NUM_POINTS / 2); ++i)
    {
        fprintf(SIGNAL_FILE, "%18.9f %18.9f\n", (double)i / (double)fs, signal[i]);
        fprintf(FREQUENCIS, "%18.9f %18.9f\n", freq[i], freqAmplitudes[i]);
    }
    fclose(SIGNAL_FILE);
    fclose(FREQUENCIS);

    return 0;
}

void make_signal(dim1& signal, const int fs)
{
    /* 
     * Generate two sine waves of different frequencies and amplitudes.
     */

    int i;
    double dt = 1.0 / fs;
    unsigned NUM_POINTS = signal.size();

    for (i = 0; i < NUM_POINTS; ++i)
    {
        double theta = (double)i / (double)fs;
        signal[i] = 1.0 * sin(50.25 * 2.0 * M_PI * theta) +
                    0.5 * sin(80.50 * 2.0 * M_PI * theta);
    }
}

void fft(const dim1 &signal, dim1& freq, dim1 &freqAmplitudes,  const double dt)
{
    const int NUM_POINTS = signal.size();
    const double fs = 1.0 / (dt + 0.0);
    double *signalArray = new double[NUM_POINTS];
    unsigned flags{0};

    fftw_complex result[NUM_POINTS / 2 + 1];
    fftw_plan plan = fftw_plan_dft_r2c_1d(NUM_POINTS,
                                          signalArray,
                                          result,
                                          flags);
    for (int i = 0; i < NUM_POINTS; i++)
        signalArray[i] = signal[i];

    fftw_execute(plan);
    for (int i = 0; i < (0.5 * NUM_POINTS); i++)
    {
        freqAmplitudes[i] = 2.0 / (double)(NUM_POINTS) * sqrt(result[i][REAL] * result[i][REAL] + result[i][IMAG] * result[i][IMAG]);
        freq[i] = i / double(NUM_POINTS) * fs;
    }

    fftw_destroy_plan(plan);
    delete[] signalArray;
}
