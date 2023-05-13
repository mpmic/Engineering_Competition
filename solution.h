//! Rohde & Schwarz Engineering Competition 2023
//!
//! This is the code to speed up. Enjoy!

#pragma once

#include "ec2023/ec2023.h"
#include <iostream>      
#include <iomanip>
#include <vector>

static constexpr float OVERLAP_RATIO = 0.75;
static constexpr size_t WINDOW_SIZE = 1024;

void compute_fourier_transform(const std::vector<ec::Float>& input, std::vector<ec::Float>& outputReal, std::vector<ec::Float>& outputImag);

std::vector<ec::Float> process_signal(const std::vector<ec::Float>& inputSignal)
{  
  const size_t numSamples = inputSignal.size();
  const size_t sizeSpectrum = (WINDOW_SIZE / 2) + 1;
  const size_t stepBetweenWins = static_cast<size_t>(ceil(WINDOW_SIZE * (1 - OVERLAP_RATIO)));
  const size_t numWins = (numSamples - WINDOW_SIZE) / stepBetweenWins + 1;
  const ec::Float PI = 3.14159265358979323846f;

  std::vector<ec::Float> signalWindow(WINDOW_SIZE);
  std::vector<ec::Float> signalFreqReal(WINDOW_SIZE);
  std::vector<ec::Float> signalFreqImag(WINDOW_SIZE);
  std::vector<ec::Float> spectrumWindow(sizeSpectrum);
  std::vector<ec::Float> outputSpectrum(sizeSpectrum, std::numeric_limits<float>::lowest());

  size_t idxStartWin = 0;
  ec::Float blackmanWinCoef;
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
        vecHw.mul32(ii * 32, ec::Float(1/(WINDOW_SIZE - 1)), ii * 32, 32ull);

        if (ii < (size_t)(WINDOW_SIZE / 32)) {
            vecHw.mul32(ii * 32, ec::Float(2.0f), ii * 32, 32ull);  
            vecHw.mul32(WINDOW_SIZE + ii * 32, ec::Float(4.0f), WINDOW_SIZE + ii * 32, 32ULL);
        }
    }

    for (size_t ii = 0; ii < (size_t)(WINDOW_SIZE / 4); ii++) {
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

      // blackmanWinCoef = 0.42f - 0.5f * ec_cos(ec::Float(I) * 2.0f * PI / (WINDOW_SIZE - 1));                                         
      // blackmanWinCoef = blackmanWinCoef + 0.08f * ec_cos(ec::Float(I) * 4.0f * PI / (WINDOW_SIZE - 1));

      // signalWindow[I] = inputSignal[I + idxStartWin] * blackmanWinCoef;

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

void compute_fourier_transform(const std::vector<ec::Float>& input, std::vector<ec::Float>& outputReal, std::vector<ec::Float>& outputImag)
{
  const ec::Float PI = 3.14159265358979323846f;

  size_t inputSize = input.size();

  outputReal.clear();
  outputReal.resize(inputSize, 0.0f);
  outputImag.clear();
  outputImag.resize(inputSize, 0.0f);

  for (size_t I = 0; I < inputSize; ++I)
  {
    for (size_t J = 0; J < inputSize; ++J)
    {
      const ec::Float angleTerm = (-2.0f * PI) * ec::Float(I) * J * (1.0f / ec::Float(inputSize));

      outputReal[I] += input[J] * ec_cos(angleTerm);
      outputImag[I] += input[J] * ec_sin(angleTerm);
    }
  }

  return;
}

