//! Rohde & Schwarz Engineering Competition 2023
//!
//! This is the code to speed up. Enjoy!
//! This code performs Fourier Transform on a given input signal to obtain the spectrum.
//! It uses the Cooley-Tukey algorithm for Fast Fourier Transform (FFT) in a divide-and-conquer manner.
//! The spectrum is computed for multiple windows of the input signal, and the maximum value for each frequency bin is recorded.
//! A Blackman window is applied to each signal window before computing the FFT.
//! The final output is a vector representing the spectrum of the input signal.

#pragma once

#include "ec2023/ec2023.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <valarray>

static const ec::Float PI = ec::Float(3.14159265358979323846f); // Define the constant PI
static constexpr float OVERLAP_RATIO = 0.75; // Define the overlap ratio for windowing
static constexpr size_t WINDOW_SIZE = 1024; // Define the size of each window

typedef std::complex<ec::Float> Complex; // Define the complex number type
typedef std::valarray<Complex> CArray; // Define the valarray of complex numbers type

// Cooley–Tukey FFT (in-place, divide-and-conquer)
void fft(CArray& x)
{
  const size_t N = x.size();
  // Base case: if the size is 1 or less, no further computation is needed
  if (N <= 1) return;

  // Split the input array into even-indexed elements
  CArray even = x[std::slice(0, N/2, 2)];
  // Split the input array into odd-indexed elements
  CArray  odd = x[std::slice(1, N/2, 2)];

  // Recursively compute the FFT on the even-indexed elements
  fft(even);
	//  Recursively compute the FFT on the odd-indexed elements
  fft(odd);

  for (size_t k = 0; k < N/2; ++k)
  {
	// Compute the twiddle factor for the kth frequency bin
	ec::Float real_part = ec::Float(1.0f) * ec::ec_cos(-2.0f * PI * ec::Float(k) / ec::Float(N));
	ec::Float imag_part = ec::Float(1.0f) * ec::ec_sin(-2.0f * PI * ec::Float(k) / ec::Float(N));

	// Apply the twiddle factor to the odd-indexed element
	Complex t = Complex(real_part, imag_part) * odd[k];

	// Combine the even-indexed element with the transformed odd-indexed element
	x[k    ] = even[k] + t;
	// Combine the even-indexed element with the conjugate of the transformed odd-indexed element
	x[k+N/2] = even[k] - t;
  }
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

// Function to process the input signal and compute the spectrum
std::vector<ec::Float> process_signal(const std::vector<ec::Float>& inputSignal)
{
  const size_t numSamples = inputSignal.size();

  // Compute the size of the spectrum
  const size_t sizeSpectrum = (WINDOW_SIZE / 2) + 1;

  // Compute the step size between windows
  const size_t stepBetweenWins = static_cast<size_t>(ceil(WINDOW_SIZE * (1 - OVERLAP_RATIO)));

  // Compute the number of windows
  const size_t numWins = (numSamples - WINDOW_SIZE) / stepBetweenWins + 1;

  // Create a vector to store each signal window
  std::vector<ec::Float> signalWindow(WINDOW_SIZE);

  // Create vectors to store the real and imaginary parts of the transformed signal
  std::vector<ec::Float> signalFreqReal(WINDOW_SIZE);
  std::vector<ec::Float> signalFreqImag(WINDOW_SIZE);

  // Create a vector to store the spectrum for each window
  std::vector<ec::Float> spectrumWindow(sizeSpectrum);

  // Create a vector to store the final output spectrum, initialized with the lowest possible values
  std::vector<ec::Float> outputSpectrum(sizeSpectrum);

  // Create a vector to store the Blackman window coefficients
  std::vector<ec::Float> blackmanWinCoef(WINDOW_SIZE);
  for (size_t I = 0; I < WINDOW_SIZE; I++)
  {
	// Compute the Blackman window coefficient for each sample
	blackmanWinCoef[I] = 0.42f - 0.5f * ec_cos(ec::Float(I) * 2.0f * PI / (WINDOW_SIZE - 1));
	blackmanWinCoef[I] += 0.08f * ec_cos(ec::Float(I) * 4.0f * PI / (WINDOW_SIZE - 1));
  }

  // Initialize the starting index of the current window
  size_t idxStartWin = 0;

  for (size_t J = 0; J < numWins; J++)
  {
	for (size_t I = 0; I < WINDOW_SIZE; I++)
	{
	  // Apply the Blackman window to each sample of the current window
	  signalWindow[I] = inputSignal[I + idxStartWin] * blackmanWinCoef[I];
	}

	compute_fourier_transform(signalWindow, signalFreqReal, signalFreqImag);

	for (size_t I = 0; I < sizeSpectrum; I++)
	{
	  ec::Float freqVal = signalFreqReal[I] * signalFreqReal[I] + signalFreqImag[I] * signalFreqImag[I];

	  // Take the square root to obtain the magnitude
	  freqVal = ec_sqrt(freqVal);

	  // Normalize the magnitude by the window size
	  freqVal = freqVal / ec::Float(WINDOW_SIZE);


	  // Scale the magnitude by a factor of 2 for non-zero and non-DC frequency bins
	  if (I > 0 && I < sizeSpectrum - 1) freqVal = freqVal * 2.0f;

	  // Square the magnitude to obtain the power spectrum
	  freqVal = freqVal * freqVal;

	  // Convert the power spectrum to decibels
	  freqVal = 10.0f * ec_log10(1000.0f * freqVal);

	  // Update the output spectrum by taking the maximum value for each frequency bin
	  outputSpectrum[I] = ec_max(outputSpectrum[I], freqVal);
	}

	// Move the starting index to the next window
	idxStartWin += stepBetweenWins;

  }

  return outputSpectrum;
}

