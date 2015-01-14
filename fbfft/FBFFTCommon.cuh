// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "cuda/Complex.cuh"
#include "cuda/ComputeCapabilities.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"
#include "cuda/fbfft/Twiddles.cuh"

#include <cuda_runtime.h>
#include <glog/logging.h>

using namespace facebook::cuda;

namespace facebook { namespace cuda { namespace fbfft {

template <typename T>
__device__ __host__ T numHermitian(T commonCols) {
  return commonCols / 2 + 1;
}

namespace detail {

#define PI 3.14159265358979323846264338327f

__device__ __forceinline__
unsigned int reverse(unsigned int x, unsigned int nbits) {
  return __brev(x) >> (WARP_SIZE - nbits);
}

// This adjustment modulo FFTSize is used as a stepping stone to cram multiple
// FFTs of size < WARP_SIZE into a single warp.
// The invariant is:
//   assert(FFTPerWarp * FFTSize == blockDim.x || FFTPerWarp == 1);
// This has no effect if FFTSize >= WARP_SIZE or FFTPerWarp == 1.
// This is for the cases 2, 4, 8 and 16 and buys us additional perf.
template <int FFTSize>
__device__ __forceinline__ int adjustedThreadIdxX() {
  if (FFTSize < WARP_SIZE) {
    return (threadIdx.x & (FFTSize - 1));
  } else {
    return threadIdx.x;
  }
}

template <int FFTSize>
__device__ __forceinline__ int adjustedThreadIdxY() {
  if (FFTSize < WARP_SIZE) {
    return (threadIdx.y & (FFTSize - 1));
  } else {
    return threadIdx.y;
  }
}

// Computes the batch number based on the fact that batches are divided by:
//   - blockIdx.x, each block computes a chunk of bacthes,
//   - threadIdx.z, each z dimensions computes a subchcunk of batches to
//     increase occupancy,
//   - exactly FFTPerWarp FFTs are processed by one warp
// These 3 subdivisions interact to compute the actual batch size.
template <int FFTSize, int FFTPerWarp>
__device__ __forceinline__ int adjustedBatch() {
  if (FFTSize < WARP_SIZE) {
    int LogFFTSize = getMSB<FFTSize>();
    int LogFFTPerWarp = getMSB<FFTPerWarp>();
    return (threadIdx.x >> LogFFTSize) +
      (blockIdx.x << LogFFTPerWarp) +
      ((threadIdx.z * gridDim.x) << LogFFTPerWarp);
  } else {
    return blockIdx.x + threadIdx.z * gridDim.x;
  }
}

// Computes the batch number based on the fact that batches are divided by:
//   - blockIdx.x, each block computes a chunk of batches,
//   - threadIdx.z, each z dimensions computes a subchunk of batches to
//     increase occupancy,
//   - exactly FFTPerWarp FFTs are processed by one warp
// These 3 subdivisions interact to compute the actual batch size.
// In the R2C case, we additionally compute 2 real FFTs as a single complex FFT
template <int FFTSize, int FFTPerWarp, bool ForwardFFT>
__device__ __forceinline__ int adjustedBatchR2C() {
  if (FFTSize < WARP_SIZE) {
    int LogFFTSize = getMSB<FFTSize>();
    int LogFFTPerWarp = getMSB<FFTPerWarp>();
    return 2 * ((threadIdx.x >> LogFFTSize) +
                (blockIdx.x << LogFFTPerWarp) +
                ((threadIdx.z * gridDim.x) << LogFFTPerWarp));
  } else {
    return 2 * (blockIdx.x + threadIdx.z * gridDim.x);
  }
}

template <int FFTSize>
struct FFT1DCoeffs {
  enum {
    ColumnsPerWarp = (FFTSize + WARP_SIZE - 1) / WARP_SIZE
  };
  __device__ __forceinline__ Complex& operator[](int i) {
    return coeff[i];
  }
  __device__ __forceinline__ Complex operator[](int i) const {
    return coeff[i];
  }

  Complex coeff[ColumnsPerWarp];
};


__device__ __forceinline__ Complex ldg(const Complex* p) {
  return Complex(__ldg(&(p->re())),
                 __ldg(&(p->im()))
                );
}

template <int FFTSize>
struct FFT1DRoots : public FFT1DCoeffs<FFTSize> {
  // Computes the twiddles for the least amount possible of registers and uses
  // trigonometric symmetries to populate the other registers.
  // We always compute at least 1 warpful of value using cexp.
  // For FFTs <= WARP_SIZE we are done
  // For FFTs >= WARP_SIZE, given the number of registers per warp we know which
  // register indices fall at PI/4, PI/2, PI and 2*PI.
  // Since we always compute at least 1 warpful of values, we only consider
  // exact subdivisions of WARP_SIZE for symmetries.
  // For instance:
  //   - for FFTSize == 64, we have 2 registers corresponding to each half of
  //     the unit circle. We compute the first register (and not less by
  //     construction) and then we can use symmetry wrt -PI to fill the other
  //     register.
  //   - for FFTSize == 128, we have 4 registers corresponding to each
  //     quadrant of the unit circle. We compute the first register (and not
  //     less by construction) and then we can use symmetry wrt -PI/2 and -PI
  //     to fill the other registers.
  //
  // This is critical performance-wise and works well atm with unrolling.
  //
  // Twiddles are more efficiently computed for 1D FFTs and more efficiently
  // loaded from constant memory for 2D FFTs.
  template <bool ForwardFFT>
  __device__ __forceinline__ void twiddles() {
    // These are the sizes empirically determined to be more SFU bound
    if ((ForwardFFT && (FFTSize == 16 || FFTSize == 32)) ||
        (!ForwardFFT && (FFTSize == 128))) {
      twiddlesFromMemory<ForwardFFT>();
      return;
    }

    const float twoPi = (ForwardFFT) ? -2.0f * PI : 2.0f * PI;
    // Note that we ever only need half the twiddles; see ASCII diagram:
    // for FFT-256 we only use w^0 .. w^127 and then recursively only halves.
    if (this->ColumnsPerWarp >= 4) {
#pragma unroll
for (int index = 0; index < ceil((int)this->ColumnsPerWarp, 2); ++index) {
        // Can always use adjustedThreadIdxX since blockDim.x == WARP_SIZE
        // is enforced
        int x = adjustedThreadIdxX<FFTSize>() + index * WARP_SIZE;
        if (index < ceil((int)this->ColumnsPerWarp, 4)) {
          // Compute in any case
          (*this)[index].cexp(twoPi * (1.0f / (float)FFTSize) * x);
        } else if (index < ceil((int)this->ColumnsPerWarp, 2)) {
          if (ForwardFFT) {
            // Symmetry wrt -PI/2
            (*this)[index] =
              (*this)[index - ceil((int)this->ColumnsPerWarp, 4)]
              .transpose()
              .conjugate();
          } else {
            // Symmetry wrt PI/2
            (*this)[index] =
              - (*this)[index - ceil((int)this->ColumnsPerWarp, 4)]
              .transpose()
              .conjugate();
          }
        } else {
          // Symmetry wrt -PI == PI
          (*this)[index] = -(*this)[this->ColumnsPerWarp - index];
        }
      }
    } else if (this->ColumnsPerWarp == 2) {
      // Compute in any case, can always use adjustedThreadIdxX since
      // blockDim.x == WARP_SIZE is enforced
      int x = adjustedThreadIdxX<FFTSize>();
      (*this)[0].cexp(twoPi * (1.0f / (float)FFTSize) * x);
      // Symmetry wrt -PI, skip since only need half
    } else {
      // Compute in any case
      // adjustedThreadIdxX<FFTSize>() lets us cram multiple < WARP_SIZE FFTs in
      // a warp
      int x = adjustedThreadIdxX<FFTSize>();
      (*this)[0].cexp(twoPi * (1.0f / (float)FFTSize) * x);
    }
  }

  // This gets another:
  //   10% performance for 2d 16x16
  //   20% performance for 2d 32x32
  // However it performs worse in all the other cases, still 16x16 and 32x32
  //   are important enough cases that we want the absolute best perf for them.
  template <bool ForwardFFT>
  __device__ __forceinline__ void twiddlesFromMemory() {
#pragma unroll
    for (int index = 0; index < ceil((int)this->ColumnsPerWarp, 2); ++index) {
      int x = threadIdx.x % FFTSize + index * WARP_SIZE;
      (*this)[index] = (ForwardFFT)  ?
        ldg(&((Complex*)twiddleFactors)[x * (kNumTwiddles / FFTSize)]) :
        ldg(&((Complex*)twiddleFactors)[x * (kNumTwiddles / FFTSize)]).
        conjugate();
    }
  }
};

template <int FFTSize>
struct FFT1DBitReversal {
  enum {
    ColumnsPerWarp = (FFTSize + WARP_SIZE - 1) / WARP_SIZE
  };
  __device__ __forceinline__ int& operator[](int i) {
    return bitReversed[i];
  }
  __device__ __forceinline__ int operator[](int i) const {
    return bitReversed[i];
  }

  __device__ __forceinline__ void computeBitReversal(const int index) {
    int LogFFTSize = cuda::getMSB<FFTSize>();
    int x = adjustedThreadIdxX<FFTSize>() + index * blockDim.x;
    bitReversed[index] = reverse(x, LogFFTSize);
  }

  int bitReversed[ColumnsPerWarp];
};

// Pure within a warp reversal for FFT sizes <= WARP_SIZE.
// For sizes >= 64 this is trickier since we need a cross-register,
// cross-warp bit reversal.
// Can be done inefficiently with a loop or local memory.
// Q: How can we make sure it will always unroll statically ?
// A: Just use shared memory for the bit reversal portion, it will only
// consume 2 * FFTSize floats per block.
template <int FFTSize, int FFTPerWarp>
 __device__ __forceinline__
void bitReverse1DWarp(FFT1DCoeffs<FFTSize>& coeffs,
                      const FFT1DBitReversal<FFTSize>& bits,
                      const int index) {
  assert(coeffs.ColumnsPerWarp == 1);
  assert(index == 0);
  assert(FFTSize <= WARP_SIZE);

  // Only reverse and permute within blockDim.x boundary which allows to cram
  // multiple FFTs smaller than WARP_SIZE into a single warp
  int LogFFTPerWarp = cuda::getMSB<FFTPerWarp>();
  coeffs[index] = shfl(coeffs[index],
                       bits[index],
                       blockDim.x >> LogFFTPerWarp);
}

// Helper function useful for maintaining the twiddle factor distribution
// invariant. Assuming registers r1 and r2, distributed across warps,
// we write r1[0, ... 31] and r2[0, ... 31].
// This concatenates r1 | r2 and keeps only the entries from the even warps.
// r1 and r2 both contain these values on exit.
// This is useful for simplifying the distribution of twiddle factors.
//
// Consider the case FFT-128, by construction:
//   r1[0, .. 31] == r3[0, .. 31] = [w^0 , .. w^31]
//   r2[0, .. 31] == r4[0, .. 31] = [w^32, .. w^63]
//
// After selectEvenWarpDistributed, all registers are equal and we have:
//   r1[0, .. 31] == ... == r4[0, .. 31] == [w^0, w^2, .. w^62]
//
// This occurs one more time to obtain:
//   r1[0, .. 31] == ... == r4[0, .. 31] == [w^0, w^4, .. w^60, 16 x garbage]
//
// The garbage is never read in decimateInFrequency1DWarp.
//
// Formally:
// r1[k] <- concat(r1, r2) [2k] for k \in [0 .. WARP_SIZE - 1]
// r2 <- r1
//
__device__ __forceinline__
void selectEvenWarpDistributed(Complex& r1, Complex& r2) {
  // E.g. stating from:
  //   r1[w^0, w^1, ... w^31] and r2[w^32, w^33, ...w^63]
  //
  // Set
  //   r1[w^0 , w^2 , ... w^30 |         16 x garbage]
  //   r2[16 x garbage         | w^32, w^34, ... w^62]
  //
  // And merge into:
  //   r1[w^0 , w^2 , ... w^30 | w^32, w^34, ... w^62]
  //
  // Dark compiler magic: trying to reduce this down to Complex loses 10%
  // perf. This seems related to instruction mix, divergence and the compiler
  // not able to reorder instructions past divergent points (which is
  // reasonable).
  r1.re() = shfl(r1.re(), 2 * getLaneId());
  r2.re() = shfl(r2.re(), 2 * getLaneId() - WARP_SIZE);
  if (threadIdx.x >= HALF_WARP_SIZE) {
    r1.re() = r2.re();
  }
  r1.im() = shfl(r1.im(), 2 * getLaneId());
  r2.im() = shfl(r2.im(), 2 * getLaneId() - WARP_SIZE);
  if (threadIdx.x >= HALF_WARP_SIZE) {
    r1.im() = r2.im();
  }
  r2 = r1;
}

template <int FFTSize, bool ForwardFFT>
__device__ __forceinline__ void load1D(const DeviceTensor<float, 2>& real,
                                       const DeviceTensor<float, 3>& complex,
                                       FFT1DCoeffs<FFTSize>& coeffs,
                                       const int batch,
                                       const int index) {
  int LogFFTSize = getMSB<FFTSize>();
  // adjustedThreadIdxX<FFTSize>() crams multiple < WARP_SIZE FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + index * blockDim.x;

  // Support zero padding without a need to copy the input data to a larger
  // array.
  // TODO: center the kernel wrt to zeros.
  // TODO: support reflection padding: pass the kernel size to fill with
  // reflection and then zero after that to pad till the FFT size.
  // TODO: support complex input (just read the imaginary part)
  // TODO: try to do something with float4 and shuffles
  if (ForwardFFT) {
    coeffs[index] =
      Complex((x < real.getSize(1)) ? real[batch][x].ldg() : 0.0f,
              0.0f);
  } else {
    coeffs[index] = (x < complex.getSize(1)) ?
      ldg(complex[batch][x].dataAs<Complex>()) :
      ldg(complex[batch][2 * (complex.getSize(1) - 1) - x].
          dataAs<Complex>()).conjugate();
  }
}

template <int FFTSize, bool ForwardFFT, bool EvenDivideBatches>
__device__ __forceinline__ void load1DR2C(const DeviceTensor<float, 2>& real,
                                          const DeviceTensor<float, 3>& complex,
                                          FFT1DCoeffs<FFTSize>& coeffs,
                                          const int batch,
                                          const int index) {
  int LogFFTSize = getMSB<FFTSize>();
  // adjustedThreadIdxX<FFTSize>() crams multiple < WARP_SIZE FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + index * blockDim.x;

  // Support zero padding without a need to copy the input data to a larger
  // array.
  // TODO: center the kernel wrt to zeros.
  // TODO: support reflection padding: pass the kernel size to fill with
  // reflection and then zero after that to pad till the FFT size.
  if (ForwardFFT) {
    // R2C
    coeffs[index] = (x < real.getSize(1)) ?
      // y = x1 + i. x2
      Complex(real[batch][x],
              (EvenDivideBatches || batch + 1 < complex.getSize(0)) ?
              real[batch + 1][x] : 0.0f) :
      Complex(0.0f);
  } else {
    // C2R
    Complex tmp1 = (x < complex.getSize(1)) ?
      *(complex[batch][x].dataAs<Complex>())
      :
      complex[batch][2 * (complex.getSize(1) - 1) - x].
      dataAs<Complex>()->conjugate();
    Complex tmp2 =
      (EvenDivideBatches || batch + 1 < complex.getSize(0)) ?
      ((x < complex.getSize(1)) ?
       *(complex[batch + 1][x].dataAs<Complex>())
       :
       complex[batch + 1][2 * (complex.getSize(1) - 1) - x].
       dataAs<Complex>()->conjugate())
      :
      Complex(0.0f);
    // y = x1 + i. x2
    coeffs[index] = Complex(tmp1.re() - tmp2.im(),
                            tmp1.im() + tmp2.re());
  }
}

template <int FFTSize, bool ForwardFFT>
__device__ __forceinline__ void store1D(DeviceTensor<float, 2>& real,
                                        DeviceTensor<float, 3>& complex,
                                        const FFT1DCoeffs<FFTSize>& coeffs,
                                        const int batch,
                                        const int index) {
  // adjustedThreadIdxX<FFTSize>() crams multiple < WARP_SIZE FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + index * blockDim.x;
  if (ForwardFFT && x < complex.getSize(1)) {
    // TODO: try to do something with float4 and shuffles
    complex[batch][x][0].as<Complex>() = coeffs[index];
  } else if (x < real.getSize(1)) {
    // TODO: try to do something with float4 and shuffles
    real[batch][x] = coeffs[index].re();
  }
}

template <int FFTSize>
__device__ __forceinline__ const Complex HermitianModuloCoefficient(
    const FFT1DCoeffs<FFTSize>& coeffs, int index) {
  assert(FFTSize > 32);
  // This monstrosisty below is unfortunately necessary to recover the
  // proper index from (N - m) % N.
  // As is, it results in local memory spilling.
  // return ((FFTSize - (index * WARP_SIZE + threadIdx.x)) % FFTSize) /
  //        WARP_SIZE;

  // After unrolling by hand, it turns out it can be expressed as follows.
  return (threadIdx.x == 0) ?
    coeffs[(coeffs.ColumnsPerWarp - index) % coeffs.ColumnsPerWarp] :
    coeffs[coeffs.ColumnsPerWarp - index - 1];
}

template <int FFTSize, bool ForwardFFT, bool EvenDivideBatches>
__device__ __forceinline__ void store1DR2C(DeviceTensor<float, 2>& real,
                                           DeviceTensor<float, 3>& complex,
                                           const FFT1DCoeffs<FFTSize>& coeffs,
                                           const int batch,
                                           const int index) {
  // adjustedThreadIdxX<FFTSize>() crams multiple < WARP_SIZE FFTs in a warp
  int x = adjustedThreadIdxX<FFTSize>() + index * blockDim.x;

  // Express x[batch] = coeffs[0]{N - m}.re() + coeffs[0]{m}.re() +
  //               i . (coeffs[0]{m}.im() - coeffs[0]{N - m}.im())
  // Shfl each
  if (ForwardFFT) {
    // This is coeffs[0]{m}, other is coeffs[0]{N - m}
    Complex tmp = (FFTSize <= WARP_SIZE) ?
      coeffs[index] :
      HermitianModuloCoefficient<FFTSize>(coeffs, index);
    Complex other = (FFTSize <= WARP_SIZE) ?
      shfl(tmp,
           FFTSize - adjustedThreadIdxX<FFTSize>(),
           FFTSize) :
      shfl(tmp, (WARP_SIZE - threadIdx.x) % WARP_SIZE, WARP_SIZE);
    // Need conditional below shfl for threads participating in shfl reasons
    if (x < complex.getSize(1)) {
      Complex c1 = Complex(0.5f * (coeffs[index].re() + other.re()),
                           0.5f * (coeffs[index].im() - other.im()));
      complex[batch][x][0].as<Complex>() = c1;
    }
    if (EvenDivideBatches || batch + 1 < complex.getSize(0)) {
      // Need conditional below shfl for threads participating in shfl reasons
      if (x < complex.getSize(1)) {
        Complex c2 = Complex( 0.5f * ( coeffs[index].im() + other.im()),
                              0.5f * (-coeffs[index].re() + other.re()));
        complex[batch + 1][x][0].as<Complex>() = c2;
      }
    }
  } else if (x < real.getSize(1)) {
    real[batch][x] = coeffs[index].re();
    if (EvenDivideBatches || batch + 1 < complex.getSize(0)) {
      real[batch + 1][x] = coeffs[index].im();
    }
  }
}

template <int FFTSize>
__device__ __forceinline__
void decimateInFrequency1DWarp(Complex& coeff, Complex& root) {
  // Cannot be static due to upstream mix of function calls
  assert(FFTSize <= WARP_SIZE);

  int LogFFTSize = getMSB<FFTSize>();

#pragma unroll
  for (int logStep = 1; logStep <= LogFFTSize; ++logStep) {
    // Illustration for 1-D FFT of size 8, radix-2, decimation in frequency
    // Step 1 amongst 2,
    // Step 2 amongst 4,
    // Step 4 amongst 8,
    // ...

    Complex otherCoeff =
      shfl_xor(coeff, FFTSize >> logStep, FFTSize >> (logStep - 1));

    // Illustration for 1-D FFT of size 8, radix-2, decimation in frequency
    // Vals {1} U {3} U {5} U {7} amongst 2,
    // Vals [2, 3] U [6, 7] amongst 4,
    // Vals [4, 7] amongst 8,
    // ...
    otherCoeff = (threadIdx.x & (FFTSize >> logStep)) ?
      otherCoeff - coeff : coeff + otherCoeff;

    if (logStep < LogFFTSize) {
      // Illustration for 1-D FFT of size 8, radix-2, decimation in frequency
      // Twiddles [w^0, [w^0], w^0, [w^0], w^0, [w^0], w^0, [w^0]] amongst 2,
      // Twiddles [w^0, w^0, [w^0, w^2], w^0, w^0, [w^0, w^2]] amongst 4,
      // Twiddles [w^0, w^0, w^0, w^0, [w^0, w^1, w^2, w^3]] amongst 8,
      // ...
      int twiddleDee = (!(threadIdx.x & (FFTSize >> logStep))) ?
        0 : ((threadIdx.x & ((FFTSize >> logStep) - 1)) << (logStep - 1));
      Complex otherRoot = shfl(root, twiddleDee);
      coeff = otherCoeff * otherRoot;
    } else {
      // Last step just does radix-2 + / - which is what otherCoeff contains
      coeff = otherCoeff;
    }
  }
}

template <int FFTSize>
struct TwiddleRebalancer {
  static __device__ __forceinline__
  void rebalance(FFT1DRoots<FFTSize>&, int);
};

template <> struct TwiddleRebalancer<64> {
  static __device__ __forceinline__
  void rebalance(FFT1DRoots<64>& roots, int) {
    selectEvenWarpDistributed(roots[0], roots[1]);
  }
};

template <> struct TwiddleRebalancer<128> {
  static __device__ __forceinline__
  void rebalance(FFT1DRoots<128>& roots, int logStep) {
    if (logStep == 1) {
      selectEvenWarpDistributed(roots[0], roots[1]);
      selectEvenWarpDistributed(roots[2], roots[3]);
      roots[1] = roots[2];
      roots[2] = roots[0];
    } else {
      assert(logStep == 2);
      selectEvenWarpDistributed(roots[0], roots[1]);
      roots[2] = roots[0];
      roots[3] = roots[0];
    }
  }
};

template <> struct TwiddleRebalancer<256> {
  static __device__ __forceinline__
  void rebalance(FFT1DRoots<256>& roots, int logStep) {
    if (logStep == 1) {
      selectEvenWarpDistributed(roots[0], roots[1]);
      selectEvenWarpDistributed(roots[2], roots[3]);
      selectEvenWarpDistributed(roots[4], roots[5]);
      selectEvenWarpDistributed(roots[6], roots[7]);
      roots[1] = roots[2];
      roots[2] = roots[4];
      roots[3] = roots[6];

      roots[4] = roots[0];
      roots[5] = roots[1];
      roots[6] = roots[2];
      roots[7] = roots[3];
    } else if (logStep == 2) {
      assert(logStep == 2);
      selectEvenWarpDistributed(roots[0], roots[1]);
      selectEvenWarpDistributed(roots[2], roots[3]);

      roots[1] = roots[2];

      roots[2] = roots[0];
      roots[3] = roots[1];

      roots[4] = roots[0];
      roots[5] = roots[1];
      roots[6] = roots[0];
      roots[7] = roots[1];
    } else {
      assert(logStep == 3);
      selectEvenWarpDistributed(roots[0], roots[1]);

      roots[1] = roots[0];

      roots[2] = roots[0];
      roots[3] = roots[0];

      roots[4] = roots[0];
      roots[5] = roots[0];
      roots[6] = roots[0];
      roots[7] = roots[0];
    }
  }
};

// The following ASCII shows the breakdown of a 1-D FFT-256 into
// the size 128 and 64-steps.
// Each 64 step is followed by 2 32-steps.
// A 32 step is the granularity of distributed storage (each warp holding 1
// value per 32-step).
// At this granularity, communication is exclusively across registers.
// Twiddle factors are continuously readjusted at each step.
// |-------|                |-------|
// | Reg0  |                | Reg0  |
// |       |                |-------|
// |-------|                | Reg1  |
// | Reg1  |                |-------|
// |-------|                |-------| w^0
// | Reg2  |                | Reg2  |  .
// |-------|                |-------|  .
// | Reg3  |                | Reg3  |  .
// |-------|                |-------| w^126 (increment 2)
//
// |-------|  w^0           |-------|
// | Reg4  |                | Reg4  |
// |       |                |-------|
// |-------|                | Reg5  |
// | Reg5  |   .            |-------|
// |-------|   .            |-------| w^0
// | Reg6  |   .            | Reg6  |  .
// |-------|                |-------|  .
// | Reg7  |                | Reg7  |  .
// |-------|  w^127 (+= 1)  |-------| w^126 (increment 2)
//
// E.g. for FFTSize = 256, we have 3 logSteps:
//   the first with 8 registers:
//     registers {{0, 4}, {1, 5}, {2, 6}, {3, 7}} communicate
//   the second with 4 registers:
//     registers {{0, 2}, {1, 3}, {4, 6}, {5, 7}} communicate
//   the third with 2 register
//     registers {{0, 1}, {2, 3}, {4, 5}, {6, 7}} communicate
//
// Note that everything is properly aligned modulo 32 and we don't need warp
// shuffles at all. The only exception may be the bit reversal phase which
// is currently implemented fully in shared memory since it would require
// fully unrolled, cross-register twiddles.
//
template <int FFTSize, int BatchUnroll, int RowsPerWarp, int RowBegin, int RowEnd>
__device__ __forceinline__
void decimateInFrequency1D(FFT1DCoeffs<FFTSize> coeffsArray[RowsPerWarp],
                           FFT1DRoots<FFTSize>& roots,
                           const int batch) {
  int LogFFTSize = getMSB<FFTSize>();
  const int kDeltaLog = LogFFTSize - LOG_WARP_SIZE;
  {
    // Computation is all within the same warp across registers.
    // Unlike shuffles, things do not update in parallel so we do have
    // WAR (a.k.a false) dependences -> need a swap temporary storage !
    // Make swap registers local to this scope
    FFT1DCoeffs<FFTSize> swap;
#pragma unroll
    for (int logStep = 1; logStep <= kDeltaLog; ++logStep) {
#pragma unroll
      for (int row = RowBegin; row < RowEnd; ++row) {
        FFT1DCoeffs<FFTSize>& coeffs = coeffsArray[row];
        assert(coeffs.ColumnsPerWarp == 1 << (LogFFTSize - LOG_WARP_SIZE));
        // Always need to process all the registers, this is not a function of
        // the logStep but only of the coeffs.ColumnsPerWarp.
        // The spacing between registers that communicate is however a function
        // of logStep.
#pragma unroll
        for (int reg = 0; reg < coeffs.ColumnsPerWarp; ++reg) {
          // By how many registers are we stepping ?
          // e.g. LogFFTSize == 8, LOG_WARP_SIZE == 5, logStep == 1 ->
          //   kDeltaLog == 3, kDeltaStep = 4
          const int kDeltaStep = (1 << (kDeltaLog - logStep));
          assert(kDeltaStep >= 0);
          assert(kDeltaStep < coeffs.ColumnsPerWarp);

          // If bit kDeltaStep is step then sub else add
          int reg2 = (reg & kDeltaStep) ? reg - kDeltaStep : reg + kDeltaStep;
          // Sanity check
          assert(reg != reg2);

          Complex otherCoeff = coeffs[reg2];
          otherCoeff = (reg > reg2) ?
            otherCoeff - coeffs[reg] : coeffs[reg] + otherCoeff;

          // Only second half requires twiddling
          if (reg > reg2) {
            // Enforce this invariant:
            //   the register is exactly reg2 and no shuffle necessary until <=
            //   WARP_SIZE
            Complex otherRoot = roots[reg2];
            // Here we could write directly to vals and not swap but performance
            // is higher writing swap, likely due to same register writing
            // across branches and predicated code generated by the compiler.
            swap.coeff[reg] = otherCoeff * otherRoot;
          } else {
            swap.coeff[reg] = otherCoeff;
          }
        }

        // Recover values from swap
#pragma unroll
        for (int reg = 0; reg < coeffs.ColumnsPerWarp; ++reg) {
          coeffs[reg] = swap.coeff[reg];
        }
      }

      // This piece of code serves the purpose of rebalancing the twiddle
      // factors across registers within a warp by merging 2 consecutive
      // registers and selecting the odd entries (effectively keeping:
      //   w^0, w^2 ... w^2*(N/2) out of w^0, w^1, ... w^N).
      // Once this is done, we have something like:
      //   w^0 .. w^62 | garbage | w^64 .. w^128 | garbage
      // That needs to be copied into:
      //   w^0 .. w^62 | w^64 .. w^128 | w^0 .. w^62 | w^64 .. w^128
      //
      // In the general case, this has a recursive behavior with log-style RAW
      // / WAR dependencies.
      // It requires full unrolling or perf will die.
      // This is what limits the FFT size to 256 atm.
      // Cannot be static due to upstream mix of function calls
      assert(WARP_SIZE <= FFTSize && FFTSize <= 256);
      // TODO: Figure out how to replace the monstruosity within
      TwiddleRebalancer<FFTSize>::rebalance(roots, logStep);
    }
  }

  // At this point we reached the FFT of WARP_SIZE, do them all in sequence
#pragma unroll
  for (int i = 0; i < (1 << kDeltaLog); ++i) {
#pragma unroll
    for (int row = RowBegin; row < RowEnd; ++row) {
      FFT1DCoeffs<FFTSize>& coeffs = coeffsArray[row];
      decimateInFrequency1DWarp<WARP_SIZE>(coeffs[i], roots[i]);
    }
  }
}

template <int FFTSize, int BatchUnroll, bool ForwardFFT, bool EvenDivideBatches>
__device__ __forceinline__
void decimateInFrequency1D(DeviceTensor<float, 2>& real,
                           DeviceTensor<float, 3>& complex,
                           FFT1DCoeffs<FFTSize> (&coeffsArray)[1],
                           const int batch) {
  // Cannot be static due to upstream mix of function calls
  assert(FFTSize >= WARP_SIZE);
  assert(blockDim.x == WARP_SIZE);

  FFT1DCoeffs<FFTSize>& coeffs = coeffsArray[0];
  FFT1DBitReversal<FFTSize> bits;

#pragma unroll
  for (int i = 0; i < coeffs.ColumnsPerWarp; ++i) {
    load1DR2C<FFTSize, ForwardFFT, EvenDivideBatches>(
      real, complex, coeffs, batch, i);
    bits.computeBitReversal(i);
  }
  FFT1DRoots<FFTSize> roots;
  roots.template twiddles<ForwardFFT>();

  decimateInFrequency1D<FFTSize, BatchUnroll, 1, 0, 1>(
    coeffsArray, roots,  batch);

  {
    // Bit reversal through shared memory because double indirection is not
    // easily unrolled.
    // TODO: see if we can use float4
    // TODO: purely in registers, starting at 256 smem already gnaws at
    // occupancy.
    // No need to sync, dependences within a single warp

    __shared__ Complex buffer[BatchUnroll][FFTSize];
    assert(blockDim.z == BatchUnroll);
#pragma unroll
    for (int reg = 0; reg < coeffs.ColumnsPerWarp; ++reg) {
      int x = getLaneId() + reg * WARP_SIZE;
      buffer[threadIdx.z][x] = coeffs[reg];
    }
    // No need to sync, dependences within a single warp
#pragma unroll
    for (int reg = 0; reg < coeffs.ColumnsPerWarp; ++reg) {
      coeffs[reg] = buffer[threadIdx.z][bits[reg]];
    }
    // No need to sync, dependences within a single warp

#pragma unroll
    for (int reg = 0; reg < coeffs.ColumnsPerWarp; ++reg) {
      store1DR2C<FFTSize, ForwardFFT, EvenDivideBatches>(
        real, complex, coeffs, batch, reg);
    }
  }
}

template <int FFTSize,
          int BatchUnroll,
          int FFTPerWarp,
          bool ForwardFFT,
          bool EvenDivideBatches>
__device__ void decimateInFrequency1DKernel(DeviceTensor<float, 2> real,
                                            DeviceTensor<float, 3> complex,
                                            int batch) {
  int LogFFTSize = getMSB<FFTSize>();
  int LogFFTPerWarp = getMSB<FFTPerWarp>();
  if (FFTSize <= WARP_SIZE) {
    FFT1DCoeffs<FFTSize> coeffs;
    load1DR2C<FFTSize, ForwardFFT, EvenDivideBatches>(
      real, complex, coeffs, batch, 0);
    FFT1DBitReversal<FFTSize> bits;
    bits.computeBitReversal(0);
    FFT1DRoots<FFTSize> roots;
    roots.template twiddles<ForwardFFT>();
    decimateInFrequency1DWarp<FFTSize>(coeffs[0], roots[0]);
    bitReverse1DWarp<FFTSize, FFTPerWarp>(coeffs, bits, 0);
    store1DR2C<FFTSize, ForwardFFT, EvenDivideBatches>(
      real, complex, coeffs, batch, 0);
  } else {
    FFT1DCoeffs<FFTSize> coeffs[1];
    decimateInFrequency1D<FFTSize, BatchUnroll, ForwardFFT, EvenDivideBatches>(
      real, complex, coeffs, batch);
  }
}


template <int FFTSize,
          int BatchUnroll,
          int FFTPerWarp,
          bool ForwardFFT>
__global__ void decimateInFrequency1DKernel(DeviceTensor<float, 2> real,
                                            DeviceTensor<float, 3> complex) {
  // Ensure proper usage of the BatchUnroll template parameter which controls
  // static shared memory allocation for bit reversals of FFTs >= 64
  // TODO: default template parameter cuda-7
  cuda_static_assert((FFTSize > WARP_SIZE && BatchUnroll >= 1) ||
                     (FFTSize <= WARP_SIZE && BatchUnroll == 1));
  cuda_static_assert(!(FFTPerWarp & (FFTPerWarp - 1)));
  cuda_static_assert(FFTPerWarp * FFTSize <= WARP_SIZE ||
                     FFTPerWarp == 1);
  assert(FFTPerWarp * FFTSize == blockDim.x || FFTPerWarp == 1);

  // Enforce that the number of FFTs we perform is divisible by the number of
  // FFTs per warp, otherwise weird divergence will occur and possibly bugs.
  assert(real.getSize(0) % FFTPerWarp == 0);
  const int batch = adjustedBatchR2C<FFTSize, FFTPerWarp, ForwardFFT>();
  if (batch >= real.getSize(0)) {
    return;
  }

  if ((FFTSize != 32 && FFTSize != 64 ) ||  // Ad-hoc but nvcc likes it
      batch + 1 >= real.getSize(0)
     ) {
    decimateInFrequency1DKernel<
      FFTSize, BatchUnroll, FFTPerWarp, ForwardFFT, false> (
        real, complex, batch);
  } else {
    decimateInFrequency1DKernel<
      FFTSize, BatchUnroll, FFTPerWarp, ForwardFFT, true> (
        real, complex, batch);
  }
}

// Performs cross warp transpose of the data in registers, synchronously for
// each register at a time and takes advantage of Hermitian symmetry.
//
// Invariants are:
//  - not synchronized on entry of the loop
//  - synchronized at each step of the loop
//  - synchronized on exit
template <int FFTSize, int SMemRows, int RowsPerWarp>
__device__ __forceinline__ void transpose2D(
      FFT1DCoeffs<FFTSize>& coeffs,
      Complex(*buffer)[SMemRows][SMemRows + 1]) {
#pragma unroll
  for (int row = 0; row < RowsPerWarp; ++row) {
#pragma unroll
    for (int reg = 0; reg < coeffs.ColumnsPerWarp; ++reg) {
      buffer[threadIdx.z][threadIdx.y][threadIdx.x] = coeffs.coeff[reg];
      __syncthreads();
      coeffs.coeff[reg] = buffer[threadIdx.z][threadIdx.x][threadIdx.y];
      __syncthreads();
    }
  }
}

// Performs cross warp transpose of the data in registers, synchronously for
// each register at a time and takes advantage of Hermitian symmetry.
//
// Supports multiple FFTs per warp.
//
// Invariants are:
//  - not synchronized on entry of the loop
//  - synchronized at each step of the loop
//  - synchronized on exit
template <int FFTSize, int SMemRows, int RowsPerWarp, int FFTPerWarp>
__device__ __forceinline__ void transpose2DMultiple(
      FFT1DCoeffs<FFTSize>& coeffs,
      Complex(*buffer)[SMemRows][SMemRows + 1]) {
  const int LogFFTSize = getMSB<FFTSize>();
  const int thx0 = (threadIdx.x >> LogFFTSize) << LogFFTSize;
#pragma unroll
  for (int row = 0; row < RowsPerWarp; ++row) {
#pragma unroll
    for (int reg = 0; reg < coeffs.ColumnsPerWarp; ++reg) {
      buffer[threadIdx.z][threadIdx.y][threadIdx.x] = coeffs.coeff[reg];
      __syncthreads();
      coeffs.coeff[reg] =
        buffer
        [threadIdx.z]
        [adjustedThreadIdxX<FFTSize>()]
        [thx0 + threadIdx.y];
      __syncthreads();
    }
  }
}

} // namespace

}}} // namespace

#include "cuda/fbfft/FBFFT-inl.cuh"
#include "cuda/fbfft/FBFFT2D-inl.cuh"
#include "cuda/fbfft/FBIFFT2D-inl.cuh"
