//! Rohde & Schwarz Engineering Competition 2023
//!
//! This is the code to speed up. Enjoy!

#pragma once

#include "ec2023/ec2023.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <forward_list>
#include <complex>
#include <iostream>
#include <valarray>

static constexpr float OVERLAP_RATIO = 0.75;
static constexpr size_t WINDOW_SIZE = 1024;
const ec::Float PI = 3.14159265358979323846f;
static const ec::Float angleTerm = (-2.0f * PI) / WINDOW_SIZE;
static std::vector<ec::Float> IJvalues;


/// [0-1023] [0 - inputSize] input/outputReal
/// [1024-2047] [inputSize - 2*inputSize] outputImag
/// [2048-2559] [2*inputSize - 2*inputSize + (inputSize / 2)] tempR
/// [2560-3071] [2*inputSize + (inputSize / 2) - 3*inputSize] tempI
/// [3072-3583] [3*inputSize - 3*inputSize + (inputSize / 2)] Tr
/// [3584 - 4096] [3*inputSize + (inputSize / 2) - 4*inputSize] Ti

enum Memory {
  outRealBegin = 0,
  outRealSize = 1024,
  outImagBegin = 1024,
  outImagSize = 1024,
  tempRBegin = 2048,
  tempRSize = 512,
  tempIBegin = 2560,
  tempISize = 512,
  TrBegin = 3072,
  TrSize = 512,
  TiBegin = 3584,
  TiSize = 512
};

// Cooley-Tukey FFT (in-place, breadth-first, decimation-in-frequency)
// Better optimized but less intuitive

// TODO: reuse dataToFifo stream fifos from multiple setups?
void setupStreamAminusB(ec::StreamHw &streamHW, size_t elementsPerPipeline, size_t numPipelines) {
  for (size_t i = 0; i < numPipelines; i++) {
	/// outReal[a]
	streamHW.startStreamDataMemToFifo(
		outRealBegin + i * elementsPerPipeline,
		4 * i + 0,
		elementsPerPipeline);
	/// outReal[b]
	streamHW.startStreamDataMemToFifo(
		outRealBegin + outRealSize / 2 + i * elementsPerPipeline,
		4 * i + 1,
		elementsPerPipeline);
	/// outImag[a]
	streamHW.startStreamDataMemToFifo(
		outImagBegin + i * elementsPerPipeline,
		4 * i + 2,
		elementsPerPipeline);
	/// outImag[b]
	streamHW.startStreamDataMemToFifo(
		outImagBegin + outImagSize / 2 + i * elementsPerPipeline,
		4 * i + 3,
		elementsPerPipeline);

	/// Real to Mem
	streamHW.startStreamDataFifoToMem(
		32 + 4 * i + 1,
		tempRBegin + i * elementsPerPipeline,
		elementsPerPipeline);

	/// Imag to Mem
	streamHW.startStreamDataFifoToMem(
		32 + 4 * i + 3,
		tempIBegin + i * elementsPerPipeline,
		elementsPerPipeline);
  }
}

void setupStreamAPlusB(ec::StreamHw &streamHW, size_t elementsPerPipeline, size_t numPipelines, size_t firstFreeFifo) {
  for (size_t i = 0; i < numPipelines; i++) {
	/// outReal[a]
	streamHW.startStreamDataMemToFifo(
		outRealBegin + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 0,
		elementsPerPipeline);
	/// outReal[b]
	streamHW.startStreamDataMemToFifo(
		outRealBegin + outRealSize / 2 + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 1,
		elementsPerPipeline);
	/// outImag[a]
	streamHW.startStreamDataMemToFifo(
		outImagBegin + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 2,
		elementsPerPipeline);
	/// outImag[b]
	streamHW.startStreamDataMemToFifo(
		outImagBegin + outImagSize / 2 + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 3,
		elementsPerPipeline);

	/// Real to Mem
	streamHW.startStreamDataFifoToMem(
		firstFreeFifo + 32 + 2 * i,
		outRealBegin + i * elementsPerPipeline,
		elementsPerPipeline);

	/// Imag to Mem
	streamHW.startStreamDataFifoToMem(
		firstFreeFifo + 32 + 2 * i + 1,
		outImagBegin + i * elementsPerPipeline,
		elementsPerPipeline);
  }
}

void
setupStreamComplexMulReal(ec::StreamHw &streamHW, size_t elementsPerPipeline, size_t numPipelines, size_t firstFreeFifo) {
  for (size_t i = 0; i < numPipelines; i++) {
	///    outputReal[b] = tempR * Tr[0][a] - tempI * Ti[0][a];
	/// tempR
	streamHW.startStreamDataMemToFifo(
		tempRBegin + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 0,
		elementsPerPipeline);
	/// tempI
	streamHW.startStreamDataMemToFifo(
		tempIBegin + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 1,
		elementsPerPipeline);
	/// Tr
	streamHW.startStreamDataMemToFifo(
		TrBegin + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 2,
		elementsPerPipeline);
	/// Ti
	streamHW.startStreamDataMemToFifo(
		TiBegin + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 3,
		elementsPerPipeline);

	/// back to Mem TODO
	streamHW.startStreamDataFifoToMem(
		firstFreeFifo + 32 + 4 * i + 3,
		outRealBegin + outRealSize / 2 + i * elementsPerPipeline,
		elementsPerPipeline);
  }
}

void
setupStreamComplexMulImag(ec::StreamHw &streamHW, size_t elementsPerPipeline, size_t numPipelines, size_t firstFreeFifo) {
  for (size_t i = 0; i < numPipelines; i++) {
	///    outputImag[b] = tempR * Ti[0][a] + tempI * Tr[0][a];
	/// tempR
	streamHW.startStreamDataMemToFifo(
		tempRBegin + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 0,
		elementsPerPipeline);
	/// tempI
	streamHW.startStreamDataMemToFifo(
		tempIBegin + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 1,
		elementsPerPipeline);
	/// Tr
	streamHW.startStreamDataMemToFifo(
		TrBegin + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 2,
		elementsPerPipeline);
	/// Ti
	streamHW.startStreamDataMemToFifo(
		TiBegin + i * elementsPerPipeline,
		firstFreeFifo + 4 * i + 3,
		elementsPerPipeline);

	/// back to Mem
	streamHW.startStreamDataFifoToMem(
		firstFreeFifo + 32 + 4 * i + 3,
		outImagBegin + outImagSize / 2 + i * elementsPerPipeline,
		elementsPerPipeline);
  }
}

void fft2(std::vector<ec::Float> &input, std::vector<ec::Float> &outputReal,
		  std::vector<ec::Float> &outputImag) {
  std::ofstream outfile;
  outfile.open("tempdata.txt");
  for (size_t i = 0; i < input.size(); i++) {
	outputReal[i] = input[i];
  }

  // DFT
  size_t inputSize = input.size();
  auto k = inputSize;
  int stepSize;
  ec::Float thetaT = PI / inputSize;


  // TODO: only 512 values need to be calculated, then we can do simple indexing :)

  // TODO: maybe we dont want to calculate with Ti/Tr vectors but rather try to multiply with constant Ti/Tr in lower k's?
  // TODO: this could save some input streams and therefore allow for more parallelism, high complexity tho
  std::vector<std::vector<ec::Float>> Ti(10);
  std::vector<std::vector<ec::Float>> Tr(10);

  auto phiT = std::complex<ec::Float>(ec::ec_cos(thetaT), -1 * ec::ec_sin(thetaT));

  for (auto j = 0; j < 10; j++) {
	Tr[j].resize(inputSize / 2);
	Ti[j].resize(inputSize / 2);
	phiT *= phiT;

	Tr[j][0] = 1.0f;
	Ti[j][0] = 0.0f;
	for (size_t i = 1; i < inputSize / 2; i++) {
	  Tr[j][i] = Tr[j][i - 1] * phiT.real() - Ti[j][i - 1] * phiT.imag();
	  Ti[j][i] = Tr[j][i - 1] * phiT.imag() + Ti[j][i - 1] * phiT.real();
	}
  }
  auto streamHW = ec::StreamHw::getSingletonStreamHw();
  streamHW->resetStreamHw();
  streamHW->createFifos(256);

  streamHW->copyToHw(input,
					 0,
					 inputSize,
					 0);
  streamHW->copyToHw(Tr[0],
					 0,
					 TrSize,
					 TrBegin);
  streamHW->copyToHw(Ti[0],
					 0,
					 TiSize,
					 TiBegin);




  ///    setup pipelines:
  ///    ec::Float tempR = outputReal[a] - outputReal[b];
  ///    ec::Float tempI = outputImag[a] - outputImag[b];
  ///
  auto nextFreeFifo = 0;

  auto numFifos = 32;
  auto maxParallelPipelines = numFifos / 4;
  auto tempRIFifoBegin = nextFreeFifo;
  for (auto i = 0; i < maxParallelPipelines; i++) {
	/// tempR
	streamHW->addOpMulToPipeline(tempRIFifoBegin + 4 * i + 1, ec::Float(-1.0f), tempRIFifoBegin + numFifos + 4 * i);
	streamHW->addOpAddToPipeline(tempRIFifoBegin + 4 * i + 0, tempRIFifoBegin + numFifos + 4 * i,
								 tempRIFifoBegin + numFifos + 4 * i + 1);
	/// tempI
	streamHW->addOpMulToPipeline(tempRIFifoBegin + 4 * i + 3, ec::Float(-1.0f),
								 tempRIFifoBegin + numFifos + 4 * i + 2);
	streamHW->addOpAddToPipeline(tempRIFifoBegin + 4 * i + 2, tempRIFifoBegin + numFifos + 4 * i + 2,
								 tempRIFifoBegin + numFifos + 4 * i + 3);
  }
  // nFF is index after tempI multiplication
  nextFreeFifo += 2 * numFifos;
  setupStreamAminusB(*streamHW, inputSize / 2 / maxParallelPipelines, maxParallelPipelines);
  streamHW->runPipeline();


  /// setup pipelines A += B
  maxParallelPipelines = numFifos / 4;
  auto aplusbFifoBegin = nextFreeFifo;
  for (auto i = 0; i < maxParallelPipelines; i++) {
	/// Real
	streamHW->addOpAddToPipeline(aplusbFifoBegin + 4 * i + 0, aplusbFifoBegin + 4 * i + 1,
								 aplusbFifoBegin + numFifos + 2 * i);
	/// Imag
	streamHW->addOpAddToPipeline(aplusbFifoBegin + 4 * i + 2, aplusbFifoBegin + 4 * i + 3,
								 aplusbFifoBegin + numFifos + 2 * i + 1);
  }
  setupStreamAPlusB(*streamHW, inputSize / 2 / maxParallelPipelines, maxParallelPipelines, aplusbFifoBegin);
  nextFreeFifo += numFifos + numFifos / 2;

  streamHW->runPipeline();

  /// setup pipelines: outputReal[b] = tempR * Tr[0][a] - tempI * Ti[0][a];
  maxParallelPipelines = numFifos / 4;
  auto complexMulRealFifoBegin = nextFreeFifo;
  for (auto i = 0; i < maxParallelPipelines; i++) {
	/// tempR * Tr[0][a]
	streamHW->addOpMulToPipeline(complexMulRealFifoBegin + 4 * i + 0,
								 complexMulRealFifoBegin + 4 * i + 2,
								 complexMulRealFifoBegin + numFifos + 4 * i + 0);
	/// tempI * Ti[0][a]
	streamHW->addOpMulToPipeline(complexMulRealFifoBegin + 4 * i + 1,
								 complexMulRealFifoBegin + 4 * i + 3,
								 complexMulRealFifoBegin + numFifos + 4 * i + 1);
	/// (tempI * Ti[0][a]) * -1
	streamHW->addOpMulToPipeline(complexMulRealFifoBegin + numFifos + 4 * i + 1,
								 ec::Float(-1.0f),
								 complexMulRealFifoBegin + numFifos + 4 * i + 2);
	/// add(subtract) left and right side together
	streamHW->addOpAddToPipeline(complexMulRealFifoBegin + numFifos + 4 * i + 0,
								 complexMulRealFifoBegin + numFifos + 4 * i + 2,
								 complexMulRealFifoBegin + numFifos + 4 * i + 3);
  }
  setupStreamComplexMulReal(*streamHW, inputSize / 2 / maxParallelPipelines, maxParallelPipelines,
							complexMulRealFifoBegin);
  nextFreeFifo += 2 * numFifos;

  streamHW->runPipeline();

  /// setup pipelines: outputImag[b] = tempR * Ti[0][a] + tempI * Tr[0][a];
  maxParallelPipelines = numFifos / 4;
  auto complexMulImagFifoBegin = nextFreeFifo;
  for (auto i = 0; i < maxParallelPipelines; i++) {
	/// tempR * Tr[0][a]
	streamHW->addOpMulToPipeline(complexMulImagFifoBegin + 4 * i + 0,
								 complexMulImagFifoBegin + 4 * i + 3,
								 complexMulImagFifoBegin + numFifos + 4 * i + 0);
	/// tempI * Ti[0][a]
	streamHW->addOpMulToPipeline(complexMulImagFifoBegin + 4 * i + 1,
								 complexMulImagFifoBegin + 4 * i + 2,
								 complexMulImagFifoBegin + numFifos + 4 * i + 1);

	/// add(subtract) left and right side together
	streamHW->addOpAddToPipeline(complexMulImagFifoBegin + numFifos + 4 * i + 0,
								 complexMulImagFifoBegin + numFifos + 4 * i + 1,
								 complexMulImagFifoBegin + numFifos + 4 * i + 3);
  }
  setupStreamComplexMulImag(*streamHW, inputSize / 2 / maxParallelPipelines, maxParallelPipelines,
							complexMulImagFifoBegin);
  nextFreeFifo += 2 * numFifos;

  streamHW->runPipeline();


  std::vector<ec::Float> outputRealStream(inputSize);
  std::vector<ec::Float> outputImagStream(inputSize);
  std::vector<ec::Float> tempRStream(inputSize / 2);
  std::vector<ec::Float> tempIStream(inputSize / 2);
  std::vector<ec::Float> tempOutA(inputSize / 2);
  std::vector<ec::Float> tempOutB(inputSize / 2);

  streamHW->copyFromHw(outputRealStream,
					   outRealBegin,
					   outRealSize,
					   0);
  streamHW->copyFromHw(outputImagStream,
					   outImagBegin,
					   outImagSize,
					   0);

  // TODO: debug
  /*streamHW->copyFromHw(tempRStream,
					   tempRBegin,
					   tempRSize,
					   0);
  streamHW->copyFromHw(tempIStream,
					   tempIBegin,
					   tempISize,
					   0);*/
  /*streamHW->copyFromHw(tempOutA,
					   3 * inputSize,
					   inputSize / 2,
					   0);

  streamHW->copyFromHw(tempOutB,
					   3 * inputSize + inputSize / 2,
					   inputSize / 2,
					   0);*/
  //k >>= 1;

  // DFT
  auto phiTr = ec::ec_cos(thetaT);
  auto phiTi = -1 * ec::ec_sin(thetaT);
  ec::Float TrScal;
  ec::Float TiScal;
  ec::Float temp;
  /*temp = phiTr * phiTr - phiTi * phiTi;
  phiTi = 2 * phiTr * phiTi;
  phiTr = temp;*/
  while (k > 1) {
	stepSize = k;
	k >>= 1;
	temp = phiTr * phiTr - phiTi * phiTi;
	phiTi = 2 * phiTr * phiTi;
	phiTr = temp;
	TrScal = 1.0f;
	TiScal = 0.0f;
	for (unsigned int l = 0; l < k; l++) {
	  for (unsigned int a = l; a < inputSize; a += stepSize) {
		unsigned int b = a + k;
		outfile << "k:" << k << " l:" << l << " a:" << a << " b:" << b << "\n";
		//outfile << k << " " << l << " " << a << " " << b << "\n";
		ec::Float tempR = outputReal[a] - outputReal[b];
		ec::Float tempI = outputImag[a] - outputImag[b];


		outputReal[a] += outputReal[b];
		outputImag[a] += outputImag[b];

		outputReal[b] = tempR * TrScal - tempI * TiScal;
		outputImag[b] = tempR * TiScal + tempI * TrScal;
		if (k == 512 && a > 120 && a < 130) {
		  std::cout << "Bad"<<std::endl;
		  /*std::cout << tempR.toFloat() << " vs bad: " << tempRStream[a].toFloat() << std::endl;*/
		}
	  }
	  temp = TrScal * phiTr - TiScal * phiTi;
	  TiScal = TrScal * phiTi + TiScal * phiTr;
	  TrScal = temp;
	}
	if (k == 512) {
	  std::cout << "debug..." << std::endl;
	  auto start = 0;
	  for (auto i = 0; i < inputSize; i++) {
		if (outputImag[i] - outputImagStream[i] > 0.001f) {
		  //std::cout << outputImag[i].toFloat() << " vs bad: " << outputImagStream[i].toFloat() << " at: " << i << std::endl;

		}
	  }
	}
  }
  // Decimate
  unsigned int m = (unsigned int) log2(inputSize);
  for (unsigned int a = 0; a < inputSize; a++) {
	unsigned int b = a;
	// Reverse bits
	b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
	b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
	b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
	b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
	b = ((b >> 16) | (b << 16)) >> (32 - m);
	if (b > a) {
	  auto tempR = outputReal[a];
	  outputReal[a] = outputReal[b];
	  outputReal[b] = tempR;

	  auto tempI = outputImag[a];
	  outputImag[a] = outputImag[b];
	  outputImag[b] = tempI;
	}
  }
}

void compute_fourier_transform(const std::vector<ec::Float> &input, std::vector<ec::Float> &outputReal,
							   std::vector<ec::Float> &outputImag, ec::Measurement &scoreMeasurement);


std::vector<ec::Float> process_signal(const std::vector<ec::Float> &inputSignal) {
  const size_t numSamples = inputSignal.size();
  const size_t sizeSpectrum = (WINDOW_SIZE / 2) + 1;
  const size_t stepBetweenWins = static_cast<size_t>(std::ceil(WINDOW_SIZE * (1 - OVERLAP_RATIO)));
  const size_t numWins = (numSamples - WINDOW_SIZE) / stepBetweenWins + 1;

  std::vector<ec::Float> signalWindow(WINDOW_SIZE);
  std::vector<ec::Float> signalFreqReal(WINDOW_SIZE);
  std::vector<ec::Float> signalFreqImag(WINDOW_SIZE);
  std::vector<ec::Float> spectrumWindow(sizeSpectrum);
  std::vector<ec::Float> outputSpectrum(sizeSpectrum, std::numeric_limits<float>::lowest());

  size_t idxStartWin = 0;

  std::vector<ec::Float> bWC(WINDOW_SIZE);
  for (auto I = 0; I < WINDOW_SIZE; I++) {
	auto innerCos = ec::Float(I) * 2.0f * PI / (WINDOW_SIZE - 1);
	bWC[I] = 0.42f - 0.5f * ec_cos(innerCos)
		+ 0.08f * ec_cos(2 * innerCos);
  }

  for (size_t J = 0; J < numWins; J++) {
	signalFreqReal.clear();
	signalFreqReal.resize(WINDOW_SIZE, 0.0f);
	signalFreqImag.clear();
	signalFreqImag.resize(WINDOW_SIZE, 0.0f);
	for (size_t I = 0; I < WINDOW_SIZE; I++) {
	  signalWindow[I] = inputSignal[I + idxStartWin] * bWC[I];
	}


	fft2(signalWindow, signalFreqReal, signalFreqImag);

	for (size_t I = 0; I < sizeSpectrum; I++) {
	  ec::Float freqVal = signalFreqReal[I] * signalFreqReal[I] + signalFreqImag[I] * signalFreqImag[I];
	  freqVal = ec_sqrt(freqVal);
	  freqVal = freqVal / ec::Float(WINDOW_SIZE);

	  if (I > 0 && I < sizeSpectrum - 1) freqVal = freqVal * 2.0f;

	  freqVal = freqVal * freqVal;

	  freqVal = 10.0f * ec_log10(1000.0f * freqVal);

	  outputSpectrum[I] = ec_max(outputSpectrum[I], freqVal);
	}

	/*for (size_t I = 0; I < sizeSpectrum; I++) {
		ec::Float freqVal = testInput[I].real() * testInput[I].real() + testInput[I].imag() * testInput[I].imag();
		freqVal = ec_sqrt(freqVal);
		freqVal = freqVal / ec::Float(WINDOW_SIZE);

		if (I > 0 && I < sizeSpectrum - 1) freqVal = freqVal * 2.0f;

		freqVal = freqVal * freqVal;

		freqVal = 10.0f * ec_log10(1000.0f * freqVal);

		outputSpectrum[I] = ec_max(outputSpectrum[I], freqVal);
	}*/

	idxStartWin += stepBetweenWins;
  }

  return outputSpectrum;
}

int littleGauss(int n) {
  return (n * n + n) / 2;
}

void compute_fourier_transform(const std::vector<ec::Float> &input, std::vector<ec::Float> &outputReal,
							   std::vector<ec::Float> &outputImag, ec::Measurement &scoreMeasurement) {
  size_t inputSize = input.size();
  outputReal.clear();
  outputReal.resize(inputSize, 0.0f);
  outputImag.clear();
  outputImag.resize(inputSize, 0.0f);

  auto streamHW = ec::StreamHw::getSingletonStreamHw();
  streamHW->resetStreamHw();

  size_t stream_width = 4096;
  size_t half_stream_width = stream_width / 2;

  std::vector<ec::Float> angleTerms(IJvalues.size());

  streamHW->resetStreamHw();

  // calculate angleterms
  const unsigned long nFifos = 64;
  const auto opWidth = stream_width; // 4096
  const auto pipelineWidth = opWidth / (nFifos / 2); // 128
  streamHW->createFifos(nFifos);

  {

	//ec::Measurement scoreMeasurement;
	const auto initscore = scoreMeasurement.calcTotalScore();

	for (unsigned long i = 0; i < nFifos / 2; i++) {
	  streamHW->addOpMulToPipeline(i, angleTerm, (nFifos / 2) + i);
	}
	for (auto counter = 0; counter * opWidth < IJvalues.size(); counter++) {
	  auto numToCalculate = std::min(opWidth, IJvalues.size() - counter * opWidth);
	  for (unsigned long i = 0; i < nFifos / 2; i++) {
		streamHW->startStreamDataMemToFifo(
			i * pipelineWidth,
			i,
			pipelineWidth);
		streamHW->startStreamDataFifoToMem(
			nFifos / 2 + i,
			i * pipelineWidth,
			pipelineWidth);
	  }

	  streamHW->resetMemTo0();
	  streamHW->copyToHw(IJvalues,
						 counter * opWidth,
						 numToCalculate,
						 0);
	  streamHW->runPipeline();
	  streamHW->copyFromHw(angleTerms,
						   0,
						   numToCalculate,
						   counter * opWidth);
	}
	std::cout << "this little endeavour cost us " << scoreMeasurement.calcTotalScore() - initscore << std::endl;
  }

  /*for(auto ij = 8192; ij < inputSize * inputSize; ij++) {
	  angleTerms[ij] = IJvalues[ij] * angleTerm;
  }*/
  //std::cout << "aT*IJ: " << angleTerms[4097].toFloat() << std::endl;
  //std::cout << "should be aT*IJ: " << IJvalues[4097].toFloat() << " * " << angleTerm.toFloat() << " = " << (IJvalues[4097] * angleTerm).toFloat() << std::endl;
  /*for (auto counter = 0; counter * opWidth < IJvalues.size(); counter++) {
	  //for (auto index = 4096; index < IJvalues.size(); index++) {
	  //angleTerms[index] = IJvalues[index] * angleTerm;

	  streamHW->resetMemTo0();
	  streamHW->copyToHw(IJvalues,
						 counter * opWidth,
						 std::min(opWidth, IJvalues.size() - counter * opWidth),
						 0);
	  streamHW->runPipeline();
	  streamHW->copyFromHw(angleTerms,
						   0,
						   std::min(opWidth, IJvalues.size() - counter * opWidth),
						   counter * opWidth);
  }*/
  /*for (auto index = inputSize; index < inputSize + 10; index++) {
	  std::cout << "angleterms after mult: " << angleTerms[index].toFloat() << std::endl;
  }*/

  std::vector<ec::Float> cosAngleTerms(angleTerms.size());
  std::vector<ec::Float> sinAngleTerms(angleTerms.size());
  for (auto i = 0; i < angleTerms.size(); i++) {
	cosAngleTerms[i] = ec::ec_cos(angleTerms[i]);
	sinAngleTerms[i] = ec::ec_sin(angleTerms[i]);
  }

  for (size_t I = 0; I < inputSize; ++I) {
	for (size_t J = 0; J < inputSize; ++J) {
	  outputReal[I] += input[J] *
		  cosAngleTerms[
			  littleGauss(
				  std::max(I, J) + 1)
				  - std::max(I, J)
				  + std::min(I, J) - 1];
	  outputImag[I] += input[J] *
		  sinAngleTerms[
			  littleGauss(
				  std::max(I, J) + 1)
				  - std::max(I, J)
				  + std::min(I, J) - 1];
	}
  }
  //std::cout << "oR sum: " << outputReal[1].toFloat() << std::endl;


  /*
  // get outputs without sum
  std::vector<ec::Float> outputRealUnsummed;

  streamed_cos(...);
  streamHW->startStreamDataMemToFifo(2048, 0, half_stream_width);
  streamHW->addOpMulToPipeline(0, 1, 2);
  streamHW->startStreamDataFifoToMem(2, 0, half_stream_width);

  for (auto counter = 0; counter * half_stream_width < IJvalues.size(); counter++) {
	  streamHW->copyToHw(angleTerms,
				   counter * half_stream_width,
						 std::min(half_stream_width, IJvalues.size() - counter * half_stream_width),
						 0);
	  streamHW->copyToHw(sortedInputs,
				   counter * half_stream_width,
						 std::min(half_stream_width, IJvalues.size() - counter * half_stream_width),
						 2048);
	  streamHW->runPipeline();
	  streamHW->copyFromHw(outputRealUnsummed,
						   0,
						   std::min(half_stream_width, angleTerms.size() - counter * half_stream_width),
					 counter * half_stream_width);
  }

  std::vector<ec::Float> outputImagUnsummed;

  streamHW->resetStreamHw();
  streamHW->createFifos(32);
  streamed_sin(...);
  streamHW->startStreamDataMemToFifo(2048, 0, half_stream_width);
  streamHW->addOpMulToPipeline(0, 1, 2);
  streamHW->startStreamDataFifoToMem(2, 0, half_stream_width);

  for (auto counter = 0; counter * half_stream_width < IJvalues.size(); counter++) {
	  streamHW->copyToHw(angleTerms,
				   counter * half_stream_width,
						 std::min(half_stream_width, IJvalues.size() - counter * half_stream_width),
						 0);
	  streamHW->copyToHw(sortedInputs,
				   counter * half_stream_width,
						 std::min(half_stream_width, IJvalues.size() - counter * half_stream_width),
						 2048);
	  streamHW->runPipeline();
	  streamHW->copyFromHw(outputRealUnsummed,
						   0,
						   std::min(half_stream_width, angleTerms.size() - counter * half_stream_width),
					 counter * half_stream_width);
  }

  for (auto counter = 0; counter * half_stream_width < IJvalues.size(); counter++) {
	  streamHW->copyToHw(angleTerms,
				   counter * half_stream_width,
						 std::min(half_stream_width, IJvalues.size() - counter * half_stream_width),
						 0);
	  streamHW->copyToHw(sortedInputs,
				   counter * half_stream_width,
						 std::min(half_stream_width, IJvalues.size() - counter * half_stream_width),
						 2048);
	  streamHW->runPipeline();
	  streamHW->copyFromHw(outputImagUnsummed,
						   0,
						   std::min(half_stream_width, angleTerms.size() - counter * half_stream_width),
					 counter * half_stream_width);
  }


  // TODO this is dumb, rather acc multiple I-rows at once and place them back, also the first 40 or so should be summed on CPU
  // TODO costfunction maybe?
  // first 64 indexes are reserved for overhang of this: (32 for the 8s and 32 for the 1s) 4*(4096 -> 128 -> 8) -> 1
  auto vecHW = ec::VecHw::getSingletonVecHw();
  auto innerMostCounter = 0;
  for (auto counter = 0; counter * (stream_width - 64) < outputRealUnsummed.size(); counter++) {
	  auto dataToCopy = std::min(stream_width - 64, outputRealUnsummed.size() - counter * (stream_width - 64));
	  vecHW->copyToHw(outputRealUnsummed,
						 counter * (stream_width - 64),
					  dataToCopy,
						 64);

	  int firstCalcAmount = std::ceil(dataToCopy / 32);
	  int secondCalcAmount = std::ceil(firstCalcAmount / 32);

	  if (dataToCopy < stream_width - 64) vecHW->resetMemTo0(dataToCopy + 64, 4096 - dataToCopy - 64);
	  for(auto acctr = 0; acctr < firstCalcAmount; acctr++) {
		  vecHW->acc32(64 + acctr * 32, 64 + acctr);
	  }

	  if (secondCalcAmount < 8) vecHW->resetMemTo0(64 + firstCalcAmount, 4096 - 64 - firstCalcAmount);
	  for(auto acctr = 0; acctr < secondCalcAmount; acctr++) {
		  vecHW->acc32(64 + acctr * 32, 32 + 8 * (counter % 4) + acctr);
	  }

	  if (secondCalcAmount < 8) vecHW->resetMemTo0(32 + secondCalcAmount, 4096 - 32 - secondCalcAmount);
	  if (counter % 4 == 3) {
		  vecHW->acc32(32, innerMostCounter++);
	  }
	  if (innerMostCounter == 31) {
		  vecHW->acc32(0, 0);
		  innerMostCounter = 1;
	  }
	  vecHW->resetMemTo0(innerMostCounter, 4096 - innerMostCounter);

	  vecHW->acc32(32, innerMostCounter++);
	  vecHW->acc32(0, 0);
  }*/


  return;
}
