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
#include "ec2023/ec_stream_hw.h"

static constexpr float PI = 3.14159265358979323846f; // Define the constant PI
static constexpr float OVERLAP_RATIO = 0.75; // Define the overlap ratio for windowing
static constexpr size_t WINDOW_SIZE = 1024; // Define the size of each window

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

typedef std::complex<ec::Float> Complex; // Define the complex number type
typedef std::valarray<Complex> CArray; // Define the valarray of complex numbers type
//

void synthfftHW(){
  ec::StreamHw& streamHW = *ec::StreamHw::getSingletonStreamHw();
  streamHW.resetStreamHw();
  streamHW.createFifos(51);
  ec::Float ecm1(-1);

  // --done on demand
  // vu 4 real lanes, 4 imag lanes [0-3, 4-7]
  //  v needs to get computed 4 times
  // w 6 real lanes, 6 imag lanes [8-13, 14- 19]
  // u 2 real lanes, 2 imag lanes [20-21, 22-23]
  //
  // -- done here
  // prepare v real two times
  // vu.real()*wr:     0 * 8 = 24
  streamHW.addOpMulToPipeline(0, 8, 24);

  // vu.real()*wr:     1 * 9 = 25
  streamHW.addOpMulToPipeline(1, 9, 25);

  // -vu.imag():       4 * [-1] = 26
  streamHW.addOpMulToPipeline(4, ecm1, 26);

  // -vu.imag():       5 * [-1] = 27
  streamHW.addOpMulToPipeline(5, ecm1, 27);

  // -vu.imag()*wi:    26 * 14 = 28
  streamHW.addOpMulToPipeline(26, 14, 28);
  // -vu.imag()*wi:    27 * 15 = 29
  streamHW.addOpMulToPipeline(27, 15, 29);

  // vr = addition calls:
  // 24 + 28 = 30 [vr]
  streamHW.addOpAddToPipeline(24, 28, 30);

  // 25 + 29 = 31 [vr]
  streamHW.addOpAddToPipeline(25, 29, 31);

  // prepare v imag two times
  // vu.real()*wi: 2 * 16 = 32
  streamHW.addOpMulToPipeline(2, 16, 32);
  // vu.real()*wi: 3 * 17 = 33
  streamHW.addOpMulToPipeline(3, 17, 33);

  // wr*vu.imag(): 10 * 6 = 34 
  streamHW.addOpMulToPipeline(10, 6, 34);
  // wr*vu.imag(): 11 * 7 = 35 
  streamHW.addOpMulToPipeline(11, 7, 35);
  // 32 + 34 = 36 [vi]
  streamHW.addOpAddToPipeline(32, 34, 36);
  // 33 + 35 = 37 [vi]
  streamHW.addOpAddToPipeline(33, 35, 37);

  // calculate the new calues s and t
  // sr = u.real() + v.real(): 20 + 30 = 38
  streamHW.addOpAddToPipeline(20, 30, 38);
  // si = u.imag() + v.imag(): 22 + 36 = 39
  streamHW.addOpAddToPipeline(22, 36, 39);
  // tr = u.real() + -v.real(): 31 * [-1] = 40; 21 + 40 = 41
  streamHW.addOpMulToPipeline(31, ecm1, 40);
  streamHW.addOpAddToPipeline(21, 40, 41);
  // ti = u.imag() + -v.imag(): 37 * [-1] = 42; 23 + 42 = 43
  streamHW.addOpMulToPipeline(37, ecm1, 42);
  streamHW.addOpAddToPipeline(23, 42, 43);

  // calculate new wr and new wi PROBLEM: needs to re synthed each iteration
  //	- because it does not copy TODO finish
  // wr = wr * wlen.real() - wi * wlen.imag(); wlen as constant
  // wr * wlen.real():	12 * [wlen.real()] = 44
  // - wi * wlen.imag() = 18 * [-1] = 45; 45 * [wlen.imag()] = 46
  //  wr = + : 44 + 46 = 47
  //
  //  wi = wr * wlen.imag() + wlen.real() * wi
  //  13 * [wlen.imag()] = 48;
  //  19 * [wlen.real()] = 49;
  //  wi: 48 + 49 = 50;

}

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
    constexpr float minusTwoPi = PI * -2.0f;
    ec::Float real_part = ec::ec_cos(minusTwoPi/ ec::Float(len));
    ec::Float imag_part = ec::ec_sin(minusTwoPi/ ec::Float(len));


    // 0  -> 7
    // real, imag, real, imag, ... 
    // real, real, real, ..., imag, imag, 
    //
    // std::vector<ec::Float> real_parts;
    // std::vector<ec::Float> imag_parts;

    //wlen twi factor
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
      //idea for pushing data into Fifo
      /*
	   *  W02 = 1
	   *  W04  
	   *  W14  
	   *  W08
	   *  W18
	   *  W28
	   *  W38
	   *  W48
      */
    }
  }
  x = tmp;
  // ec::Float wr = 1;
  // ec::Float wi = 0;

  ec::StreamHw& streamHW = *ec::StreamHw::getSingletonStreamHw();
  // streamHW.resetStreamHw();
  // Complex v(vu.real()*wr-vu.imag()*wi, vu.real()*wi+wr*vu.imag());
  // calcing the complex number v needs to get calculated twice 2x vr and vi
  // vu 4 real lanes, 4 imag lanes [0-3, 4-7]
  //  v needs to get computed 4 times
  // w 6 real lanes, 6 imag lanes [8-13, 14- 19]
  // u 2 real lanes, 2 imag lanes [20-21, 22-23]

  //how much data will be processed in parallel
  size_t sizeblock = 1;

  //real part of vu
  streamHW.startStreamDataMemToFifo(0, 0, sizeblock);
  streamHW.startStreamDataMemToFifo(0, 1, sizeblock);
  streamHW.startStreamDataMemToFifo(0, 2, sizeblock);
  streamHW.startStreamDataMemToFifo(0, 3, sizeblock);

  // for the imag part of vu
  streamHW.startStreamDataMemToFifo(sizeblock, 4, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock, 5, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock, 6, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock, 7, sizeblock);

  //w real part
  streamHW.startStreamDataMemToFifo(sizeblock*2, 8, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*2, 9, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*2, 10, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*2, 11, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*2, 12, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*2, 13, sizeblock);

  //w imag part
  streamHW.startStreamDataMemToFifo(sizeblock*3, 14, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*3, 15, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*3, 16, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*3, 17, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*3, 18, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*3, 19, sizeblock);

  //u real part
  streamHW.startStreamDataMemToFifo(sizeblock*4, 20, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*4, 21, sizeblock);

  //u imag part
  streamHW.startStreamDataMemToFifo(sizeblock*5, 22, sizeblock);
  streamHW.startStreamDataMemToFifo(sizeblock*5, 23, sizeblock);


  // space 32 - ...
  //vr calculations
  //vu.real()*wr
  //streamHW.addOpMulToPipeline(8 , wr, 32);
  //vu.imag()*wi
  //streamHW.addOpMulToPipeline(12 ,wi*(-1), 33);
  // vr = 34 [4 pipes]
  //streamHW.addOpAddToPipeline(32, 33, 34);
  //
  // vi calculation
  // vu.real()*wi
  // streamHW.addOpMulToPipeline(9 , wi, 35);
  // wr*vu.imag()
  // streamHW.addOpMulToPipeline(13 , wr, 36);
  // vr = 37 [4 pipes]
  // streamHW.addOpAddToPipeline(35 , 36, 37);
  // for one calculation already two streams/4 consumed
  // but we need 4 calculations -> rethinking, imagine vu is already precalculated to contain 
  // AGAIN -> problem 
  // Complex s(u.real() + v.real(), u.imag() + v.imag());
  // 
  //Complex t(u.real() - v.real(), u.imag() - v.imag());

  //
  // 8 *4 = 32
  //
  // 0 - 31 would be input streams
  // W02
  // - W02

  // -W04
  // W14
  // -W14




  //in the End 2 possible:
  //one layer only, but also calculate twindle factors and add them to v
  // do this
  // vu 4 real lanes, 4 imag lanes [0-3, 4-7]
  //  v needs to get computed 4 times
  // w 6 real lanes, 6 imag lanes [8-13, 14- 19]
  // u 2 real lanes, 2 imag lanes [20-21, 22-23]
  // prepare v real two times
  // vu.real()*wr:     0 * 8 = 24
  // vu.real()*wr:     1 * 9 = 25
  // -vu.imag():       4 * [-1] = 26
  // -vu.imag():       5 * [-1] = 27
  // -vu.imag()*wi:    26 * 14 = 28
  // -vu.imag()*wi:    27 * 15 = 29
  // vr = addition calls:
  // 24 + 28 = 30 [vr]
  // 25 + 29 = 31 [vr]
  //
  // prepare v imag two times
  // vu.real()*wi: 2 * 16 = 32
  // vu.real()*wi: 3 * 17 = 33
  // wr*vu.imag(): 10 * 6 = 34 
  // wr*vu.imag(): 11 * 7= 35 
  // 32 + 34 = 36 [vi]
  // 33 + 35 = 37 [vi]
  //
  // calculate the new calues s and t
  // sr = u.real() + v.real(): 20 + 30 = 38
  // si = u.imag() + v.imag(): 22 + 36 = 39
  // tr = u.real() + -v.real(): 31 * [-1] = 40; 21 + 40 = 41
  // ti = u.imag() + -v.imag(): 37 * [-1] = 42; 23 + 42 = 43
  //
  // calculate new wr and new wi
  // wr = wr * wlen.real() - wi * wlen.imag(); wlen as constant
  // wr * wlen.real():	12 * [wlen.real()] = 44
  // - wi * wlen.imag() = 18 * [-1] = 45; 45 * [wlen.imag()] = 46
  //  wr = + : 44 + 46 = 47
  //
  //  wi = wr * wlen.imag() + wlen.real() * wi
  //  13 * [wlen.imag()] = 48;
  //  19 * [wlen.real()] = 49;
  //  wi: 48 + 49 = 50;

  //second option: two layers but w needs to precomputed on the v values
  //  above solution assumes that we have them already precomputed
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

  // Create a vector to store the Blackman window coefficients
  std::vector<ec::Float> blackmanWinCoef(WINDOW_SIZE);
  auto constantBlackmanCoefficients = constantBlackmanWinCoef();

  for (size_t I = 0; I < WINDOW_SIZE; I++) {
    // Compute the Blackman window coefficient for each sample
    blackmanWinCoef[I] = constantBlackmanCoefficients[I];
  }

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

      //	  ec::Float freqVal = signalFreqReal[I] * signalFreqReal[I] + signalFreqImag[I] * signalFreqImag[I];
      ec::Float freqVal = data[I].real() * data[I].real() + data[I].imag() * data[I].imag();

      // Take the square root to obtain the magnitude
      constexpr float oneByWindowSize = (1.0f / WINDOW_SIZE);
      freqVal = ec_sqrt(freqVal) * oneByWindowSize;

      //	  // Normalize the magnitude by the window size
      //	  freqVal = freqVal * ec::Float(1 / WINDOW_SIZE);

      // Scale the magnitude by a factor of 2 for non-zero and non-DC frequency bins
      if (I > 0 && I < sizeSpectrum - 1) freqVal = freqVal * 2.0f;

      // Square the magnitude to obtain the power spectrum
      freqVal = freqVal * freqVal;

      // Convert the power spectrum to decibels
      freqVal = 10.0f * (3 + ec_log10(freqVal));

      // Update the output spectrum by taking the maximum value for each frequency bin
      outputSpectrum[I] = ec_max(outputSpectrum[I], freqVal);
    }

    // Move the starting index to the next window
    idxStartWin += stepBetweenWins;

  }

  return outputSpectrum;
}


