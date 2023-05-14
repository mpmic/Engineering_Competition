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

// typedef std::complex<ec::Float> Complex; // Define the complex number type
// typedef std::valarray<Complex> CArray; // Define the valarray of complex numbers type
// void compute_fourier_transform(const std::vector<ec::Float>& input, std::vector<ec::Float>& outputReal, std::vector<ec::Float>& outputImag);
// std::vector<ec::Float> fftCombine(std::vector<ec::Float> x1, std::vector<ec::Float> x2, size_t N);
// std::vector<ec::Float> getEvenOddTerms(std::vector<ec::Float> x, int b);
// std::vector<ec::Float> twiddle(size_t N);
// std::vector<ec::Float> fftCompute(const std::vector<ec::Float>& input, size_t N);
typedef std::complex<ec::Float> Complex; // Define the complex number type
typedef std::valarray<Complex> CArray; // Define the valarray of complex numbers type

void fft(CArray& x)
{
  const size_t N = x.size();
  if (N <= 1) return;

  CArray tmp(N);

  // Rearrange the input array using bit reversal
  for (size_t i = 0; i < N; ++i) {
	size_t j = 0;
	for (size_t bit = 0; bit < std::floor(std::log2(N)); ++bit) {
	  if (i & (1 << bit)) {
		j |= (1 << static_cast<int>(std::floor(std::log2(N)) - 1 - bit));
	  }
	}
	tmp[j] = x[i];
  }

  // In-place butterfly operations
  for (size_t len = 2; len <= N; len *= 2) {
	ec::Float real_part = ec::Float(1.0f) * ec::ec_cos(-2.0f * PI / ec::Float(len));
	ec::Float imag_part = ec::Float(1.0f) * ec::ec_sin(-2.0f * PI / ec::Float(len));
	Complex wlen(real_part, imag_part);

	for (size_t start = 0; start < N; start += len) {
	  Complex w(1);
	  for (size_t i = 0; i < len / 2; ++i) {
		Complex u = tmp[start + i];
		Complex v = tmp[start + i + len / 2] * w;
		tmp[start + i] = u + v;
		tmp[start + i + len / 2] = u - v;
		w *= wlen;
	  }
	}
  }

  x = tmp;
}


// Function to compute the Fourier transform of the input signal
void compute_fourier_transform(const std::vector<ec::Float>& input, std::vector<ec::Float>& outputReal, std::vector<ec::Float>& outputImag)
{
  size_t inputSize = input.size();

  // Create a complex valarray of ec::Float to store the input data
  CArray data(inputSize);

  for (size_t i = 0; i < inputSize; ++i)
  {
	// Convert each input sample to a complex
	data[i] = Complex(input[i], ec::Float(0.0f));

  }

  // Compute the FFT of the input data using the Cooley-Tukey algorithm
  fft(data);

  // Clear the output vectors
  outputReal.clear();

  // Resize the output vectors to match the input size
  outputReal.resize(inputSize);
  outputImag.clear();
  outputImag.resize(inputSize);

  for (size_t i = 0; i < inputSize; ++i)
  {
	// Extract the real part of the transformed data
	outputReal[i] = data[i].real();
	// Extract the imaginary part of the transformed data
	outputImag[i] = data[i].imag();
  }
}


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
