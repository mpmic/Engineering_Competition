//! Rohde & Schwarz Engineering Competition 2023
//!
//! This is the code to speed up. Enjoy!

#pragma once

#include "ec2023/ec2023.h"
#include <iomanip>
#include <vector>
#include <complex>
#include <valarray>

static constexpr float OVERLAP_RATIO = 0.75;
static constexpr size_t WINDOW_SIZE = 1024;
static const ec::Float PI = ec::Float(3.14159265358979323846f); // Define the constant PI

typedef std::complex<ec::Float> Complex; // Define the complex number type
typedef std::valarray<Complex> CArray; // Define the valarray of complex numbers type
void compute_fourier_transform(const std::vector<ec::Float>& input, std::vector<ec::Float>& outputReal, std::vector<ec::Float>& outputImag);
std::vector<ec::Float> fftCombine(std::vector<ec::Float> x1, std::vector<ec::Float> x2, size_t N);
std::vector<ec::Float> getEvenOddTerms(std::vector<ec::Float> x, int b);
std::vector<ec::Float> twiddle(size_t N);
std::vector<ec::Float> fftCompute(const std::vector<ec::Float>& input, size_t N);


std::vector<ec::Float> process_signal(const std::vector<ec::Float>& inputSignal)
{  
  const size_t numSamples = inputSignal.size();
  const size_t sizeSpectrum = (WINDOW_SIZE / 2) + 1;
  const size_t stepBetweenWins = static_cast<size_t>(ceil(WINDOW_SIZE * (1 - OVERLAP_RATIO)));
  const size_t numWins = (numSamples - WINDOW_SIZE) / stepBetweenWins + 1;
  // const ec::Float PI = 3.14159265358979323846f;

  std::vector<ec::Float> signalWindow(WINDOW_SIZE);
  std::vector<ec::Float> signalFreqReal(WINDOW_SIZE);
  std::vector<ec::Float> signalFreqImag(WINDOW_SIZE);
  std::vector<ec::Float> spectrumWindow(sizeSpectrum);
  std::vector<ec::Float> outputSpectrum(sizeSpectrum, std::numeric_limits<float>::lowest());

  size_t idxStartWin = 0;

  
  ec::VecHw& vecHw = *ec::VecHw::getSingletonVecHw();
  for (size_t J = 0; J < numWins; J++)
  {

    
    vecHw.resetMemTo0();

    for (size_t I = 0; I < WINDOW_SIZE; I++)
    {
      // signalWindow[I] = ec::Float(I) * 2.0f * PI / (WINDOW_SIZE - 1);
      signalWindow[I] = ec::Float(I);
    }

    vecHw.copyToHw(signalWindow, 0, WINDOW_SIZE, 0);
    vecHw.copyToHw(signalWindow, 0, WINDOW_SIZE, WINDOW_SIZE);

    for (size_t ii = 0; ii < (size_t)(2 * WINDOW_SIZE / 32); ii++) {
        vecHw.mul32(ii * 32, ec::Float(PI), ii * 32, 32ull);
        vecHw.mul32(ii * 32, ec::Float(ec::Float(1)/(WINDOW_SIZE - 1)), ii * 32, 32ull);

        if (ii < (size_t)(WINDOW_SIZE / 32)) {
            vecHw.mul32(ii * 32, ec::Float(2.0f), ii * 32, 32ull);  
            vecHw.mul32(WINDOW_SIZE + ii * 32, ec::Float(4.0f), WINDOW_SIZE + ii * 32, 32ULL);
        }
    }

    for (size_t ii = 0; ii < (size_t)(2 * WINDOW_SIZE / 4); ii++) {
        vecHw.cos4(ii * 4, ii * 4, 4ull);
    }

    for (size_t ii = 0; ii < (size_t)(WINDOW_SIZE / 32); ii++) {
        vecHw.mul32(ii * 32, ec::Float(-0.5f), ii * 32, 32ull);
        vecHw.mul32(WINDOW_SIZE + ii * 32, ec::Float(0.08f), WINDOW_SIZE + ii * 32, 32ull);
    }

    for (size_t ii = 0; ii < (size_t)(WINDOW_SIZE / 32); ii++) {
        vecHw.add32(ii * 32, ec::Float(0.42f), ii * 32, 32ull);
        vecHw.add32(ii * 32, WINDOW_SIZE + ii * 32, ii * 32, 32ull);
    }


    vecHw.copyToHw(inputSignal, idxStartWin, WINDOW_SIZE, WINDOW_SIZE);
    for (size_t ii = 0; ii < (size_t)(WINDOW_SIZE / 32); ii++) {
        vecHw.mul32(ii * 32, WINDOW_SIZE + ii * 32, ii * 32, 32ull);
    }

    vecHw.copyFromHw(signalWindow, 0, WINDOW_SIZE, 0);

    compute_fourier_transform(signalWindow, signalFreqReal, signalFreqImag);

    for (size_t I = 0; I < sizeSpectrum; I++)
    {
      ec::Float freqVal = signalFreqReal[I] * signalFreqReal[I] + signalFreqImag[I] * signalFreqImag[I];
      freqVal = ec_sqrt(freqVal);
      freqVal = freqVal / ec::Float(WINDOW_SIZE);

      if (I > 0 && I < sizeSpectrum - 1) freqVal = freqVal * 2.0f;

      freqVal = freqVal * freqVal;

      freqVal = 10.0f * ec_log10(1000.0f * freqVal);

      outputSpectrum[I] = ec_max(outputSpectrum[I], freqVal);
    }

    idxStartWin += stepBetweenWins;

  }

  return outputSpectrum;
}

// void compute_fourier_transform(const std::vector<ec::Float>& input, std::vector<ec::Float>& outputReal, std::vector<ec::Float>& outputImag)
// {
//   const ec::Float PI = 3.14159265358979323846f;

//   size_t inputSize = input.size();

//   outputReal.clear();
//   outputReal.resize(inputSize, 0.0f);
//   outputImag.clear();
//   outputImag.resize(inputSize, 0.0f);

//   for (size_t I = 0; I < inputSize; ++I)
//   {
//     for (size_t J = 0; J < inputSize; ++J)
//     {
//       const ec::Float angleTerm = (-2.0f * PI) * ec::Float(I) * J * (1.0f / ec::Float(inputSize));

//       outputReal[I] += input[J] * ec_cos(angleTerm);
//       outputImag[I] += input[J] * ec_sin(angleTerm);
//     }
//   }

//   return;
// }

    std::vector<ec::Float> getEvenOddTerms(std::vector<ec::Float> x, int b) {
        std::vector<ec::Float> y(x.size() / 2);
        for (int i = 0 + b; i < x.size(); i += 2) {
            y[i] = (x[i]);
        }

        return y;
    }

    void compute_fourier_transform(const std::vector<ec::Float>& input, std::vector<ec::Float>& outputReal, std::vector<ec::Float>& outputImag) {
        std::vector<ec::Float> FFT_Out = fftCompute(input, input.size());

        for (size_t ii = 0; ii < input.size(); ii++) {
            outputReal[ii] = FFT_Out[ii];
            outputImag[ii] = FFT_Out[ii + input.size()];
        }

        return;
    }

    std::vector<ec::Float> twiddle(size_t N) {
        std::vector<ec::Float> W(2 * N);
        ec::VecHw& vecHw = *ec::VecHw::getSingletonVecHw();

        vecHw.resetMemTo0();

        for (size_t ii = 0; ii < N; ii++) {
            W[ii] = ii * PI * 2 / N;
            W[ii + N] = -ii * PI * 2 / N;
            // y.push_back(complex<float>(cos(2 * M_PI * i / N), sin(2 * M_PI * (- 1) * i / N)));
        }

        vecHw.copyToHw(W, 0, 2 * N, 0);

        for (size_t ii = 0; ii < (size_t)(N / 4); ii++) {
            vecHw.cos4(ii * 4, ii * 4, 4ull);
            vecHw.sin4(ii * 4 + N, ii * 4 + N, 4ull);
        }

        vecHw.copyFromHw(W, 0, 2 * N, 0);

        return W;
    }

    std::vector<ec::Float> fftCompute(const std::vector<ec::Float>& input, size_t N) {
        if (N == 2) {
            std::vector<ec::Float> y(4);
            y[0] = (input[0] + input[1]);
            y[1] = (input[0] - input[1]);
            y[2] = (input[2] + input[3]);
            y[3] = (input[2] - input[3]);
            
            return y;
        }

        std::vector<ec::Float> tmp1, tmp2, fft1, fft2;
        tmp1 = getEvenOddTerms(input, 0);
        tmp2 = getEvenOddTerms(input, 1);
        fft1 = fftCompute(tmp1 , N/2);
        fft2 = fftCompute(tmp2 , N/2);
        return fftCombine(fft1, fft2, N);
    }


    // Marked for Improvisation
    std::vector<ec::Float> fftCombine(std::vector<ec::Float> x1, std::vector<ec::Float> x2, size_t N) {
        std::vector<ec::Float> w = twiddle(N);
        std::vector<ec::Float> y(2 * N);
                
        for (int i = 0; i < N/2; i++) {
            y[i] = (x1[i] + w[i] * x2[i] - w[i + N] * x2[i + N/2]);
            y[i + N] = (x1[i + N/2] + w[i + N] * x2[i] + w[i] * x2[i + N/2]);
        }
        for (int i = 0; i < N/2; i++) {
            y[i + N/2] = (x1[i] - (w[i + N/2] * x2[i] - w[i + N + N/2] * x2[i + N/2]));
            y[i + N + N/2] = (x1[i + N/2] - (w[i + N + N/2] * x2[i] + w[i + N/2] * x2[i + N/2]));
        }
        return y;
    }
