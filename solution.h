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
#include <cmath>
#include "ec2023/ec_stream_hw.h"

static constexpr float PI = 3.14159265358979323846f; // Define the constant PI
static constexpr float OVERLAP_RATIO = 0.75; // Define the overlap ratio for windowing
static constexpr size_t WINDOW_SIZE = 1024; // Define the size of each window
static constexpr float MINUS_TWO_PI = PI * -2.0f;
static constexpr size_t MAX_TWIDDLE_N = 1024*1024*1024;

constexpr float blackman(size_t i, size_t windowSize) {
  return 0.42f - 0.5f * std::cos(2.0f * PI * i / (windowSize - 1)) +
	  0.08f *  std::cos(4.0f * PI * i / (windowSize - 1));
}

template<std::size_t... I>
constexpr auto blackmanWinCoef(std::index_sequence<I...>) {
  return std::array<float, sizeof...(I)>{{ blackman(I, WINDOW_SIZE)... }};
}

constexpr auto constantBlackmanWinCoef() {
  return blackmanWinCoef(std::make_index_sequence<WINDOW_SIZE>{});
}

constexpr unsigned floorlog2(unsigned x)
{
  return x == 1 ? 0 : 1+floorlog2(x >> 1);
}

constexpr unsigned ceillog2(unsigned x)
{
  return x == 1 ? 0 : floorlog2(x - 1) + 1;
}

static constexpr size_t logTwiddleN = ceillog2(MAX_TWIDDLE_N);

// Compile-time computation of twiddle factors
constexpr std::array<std::complex<float>, logTwiddleN> generateTwiddleFactors() {

  std::array<std::complex<float>, logTwiddleN> twiddleFactors;

  for (size_t len = 2; len <= MAX_TWIDDLE_N; len *= 2) {

	size_t index = floorlog2(len) - 1;

	float realPart = std::cos(MINUS_TWO_PI / len);
	float imagPart = std::sin(MINUS_TWO_PI / len);
	std::complex<float> twiddleFactor(realPart, imagPart);
	twiddleFactors[index] = twiddleFactor;
  }

  return twiddleFactors;
}


typedef std::complex<ec::Float> Complex; // Define the complex number type
typedef std::valarray<Complex> CArray; // Define the valarray of complex numbers type

void fft(CArray& x)
{
  const size_t N = x.size();
  if (N <= 1) return;

  CArray rearrangedData(N);

  // Rearrange the input array using bit reversal
  for (size_t i = 0; i < N; ++i) {
	size_t j = 0;
	for (size_t bit = 0; bit < std::floor(std::log2(N)); ++bit) {
	  if (i & (1 << bit)) {
		j |= (1 << static_cast<int>(std::floor(std::log2(N)) - 1 - bit));
	  }
	}
	rearrangedData[j] = x[i];
  }

  auto preComputedTwiddleFactors = generateTwiddleFactors();
  size_t preCompIndex = 0;

  // In-place butterfly operations
  for (size_t len = 2; len <= N; len *= 2) {


	Complex twiddleFactor;

	if (len<=MAX_TWIDDLE_N){

	  twiddleFactor = preComputedTwiddleFactors[preCompIndex];
	}

	else{
	  ec::Float realPart = ec::ec_cos(MINUS_TWO_PI / ec::Float(len));
	  ec::Float imagPart = ec::ec_sin(MINUS_TWO_PI / ec::Float(len));
	twiddleFactor.real(realPart);
	twiddleFactor.real(imagPart);

	}

	for (size_t start = 0; start < N; start += len) {
	  Complex w(1);

	  for (size_t i = 0; i < len / 2; ++i) {
		Complex twiddleProduct = rearrangedData[start + i + len / 2] * w;
		rearrangedData[start + i + len / 2] = rearrangedData[start + i] - twiddleProduct;
		rearrangedData[start + i] += twiddleProduct;
		w *= twiddleFactor;
	  }

	}
	++preCompIndex;
  }

  x = std::move(rearrangedData);
}

// Function to compute the Fourier transform of the input signal
CArray compute_fourier_transform(const std::vector<ec::Float>& input)
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

  return data;
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

  // Create a vector to store the spectrum for each window
  std::vector<ec::Float> spectrumWindow(sizeSpectrum);

  // Create a vector to store the final output spectrum, initialized with the lowest possible values
  std::vector<ec::Float> outputSpectrum(sizeSpectrum);

  std::array<float, WINDOW_SIZE> blackmanWinCoef = constantBlackmanWinCoef();


  // Initialize the starting index of the current window
  size_t idxStartWin = 0;

  for (size_t J = 0; J < numWins; J++) {
	for (size_t I = 0; I < WINDOW_SIZE; I++) {
	  // Apply the Blackman window to each sample of the current window
	  signalWindow[I] = inputSignal[I + idxStartWin] * blackmanWinCoef[I];
	}

//	compute_fourier_transform(signalWindow, signalFreqReal, signalFreqImag);
	auto data = compute_fourier_transform(signalWindow);

	for (size_t I = 0; I < sizeSpectrum; I++) {

	  constexpr float oneByWindowSize = (1.0f / WINDOW_SIZE);
//	  ec::Float freqVal = signalFreqReal[I] * signalFreqReal[I] + signalFreqImag[I] * signalFreqImag[I];
	  ec::Float freqVal = (I > 0 && I < sizeSpectrum - 1) ?
		  ec_sqrt(data[I].real() * data[I].real() + data[I].imag() * data[I].imag()) * oneByWindowSize * 2.0f:
		  ec_sqrt(data[I].real() * data[I].real() + data[I].imag() * data[I].imag()) * oneByWindowSize ;

	  // Take the square root to obtain the magnitude
//	  constexpr float oneByWindowSize = (1.0f / WINDOW_SIZE);
//	  freqVal = ec_sqrt(freqVal) * oneByWindowSize;

//	  // Normalize the magnitude by the window size
//	  freqVal = freqVal * ec::Float(1 / WINDOW_SIZE);

//	  // Scale the magnitude by a factor of 2 for non-zero and non-DC frequency bins
//	  if (I > 0 && I < sizeSpectrum - 1) freqVal = freqVal * 2.0f;
//
//	  // Square the magnitude to obtain the power spectrum
//	  freqVal = freqVal * freqVal;

	  // Convert the power spectrum to decibels
//	  freqVal = 10.0f * (3 + ec_log10(freqVal));

	  // Update the output spectrum by taking the maximum value for each frequency bin
	  outputSpectrum[I] = ec_max(outputSpectrum[I],
								 10.0f * (3 + ec_log10(freqVal * freqVal)));
	}

	// Move the starting index to the next window
	idxStartWin += stepBetweenWins;

  }

  return outputSpectrum;
}


