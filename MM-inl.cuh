// Copyright 2004-present Facebook. All Rights Reserved.

#include "cuda/Complex.cuh"
#include "cuda/ComputeCapabilities.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceTensor.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <glog/logging.h>

namespace facebook { namespace cuda {

namespace detail {

__device__ __forceinline__ Complex ldg(const Complex* p) {
  return Complex(__ldg(&(p->re())),
                 __ldg(&(p->im()))
                );
}

// By construction, x * y is contiguous.
// doall xy
//   doall i, j
//     red k
//       C[i][j][x * y] += A[i][k][x * y] * B[k][j][x * y]
//
// UpdateOutput     : xy times o(b, f) <- i(b, p) . conj(f(f, p))
// AccGradParameters: xy times f(f, p) <- conj(o(b, f)) . i(b, p)
// UpdateGradInput  : xy times i(b, p) <- o(b, f) . f(f, p)
template <bool ConjugateTransposeA,
          bool ConjugateTransposeB,
          int C_XY_Placement_ThreadIdx_X,
          int C_J_Unroll,
          int C_I_Tile,
          int C_J_Tile,
          int ReductionUnroll,
          bool StaticUnrollA,
          bool StaticUnrollB,
          bool StaticUnrollCI,
          bool StaticUnrollCJ,
          bool StaticUnrollXY,
          bool StaticUnrollReduction,
          bool Accumulate>
__launch_bounds__(32 * 32, 1)
__global__ void transposeMMTiledKernel(const DeviceTensor<Complex, 3> A,
                                       const DeviceTensor<Complex, 3> B,
                                       DeviceTensor<Complex, 3> C,
                                       Complex invNorm) {
  assert(C_J_Unroll == blockDim.y);
  assert(C_XY_Placement_ThreadIdx_X == blockDim.x);
  assert(A.getSize(2) == C.getSize(2));
  assert(B.getSize(2) == C.getSize(2));
  assert(ConjugateTransposeA || A.getSize(0) == C.getSize(0));
  assert(!ConjugateTransposeA || A.getSize(1) == C.getSize(0));
  assert(ConjugateTransposeB || B.getSize(1) == C.getSize(1));
  assert(!ConjugateTransposeB || B.getSize(0) == C.getSize(1));
  assert(ConjugateTransposeA || ConjugateTransposeB ||
         A.getSize(1) == B.getSize(0));

  const int numRed =
    (ConjugateTransposeA) ? A.getSize(0) : A.getSize(1);

  const int ubi = (StaticUnrollCI) ?
    C.getSize(0) :
    ceil(C.getSize(0), (int)(C_I_Tile * gridDim.x))
    * C_I_Tile * gridDim.x;
  assert(!StaticUnrollCI || C.getSize(0) % (C_I_Tile * gridDim.x) == 0);

  const int ubj = (StaticUnrollCJ) ?
    C.getSize(1) :
    ceil(C.getSize(1), (int)(C_J_Tile * gridDim.y * blockDim.y))
    * C_J_Tile * gridDim.y * blockDim.y;
  assert(!StaticUnrollCJ ||
         C.getSize(1) % (C_J_Tile * gridDim.y * blockDim.y) == 0);

  const int ubk = (StaticUnrollReduction) ?
    numRed : ceil(numRed, ReductionUnroll) * ReductionUnroll;
  assert(!StaticUnrollReduction || numRed % ReductionUnroll == 0);

  const int NumBatches = A.getSize(2);
  const int ubxy =
    ceil(NumBatches, (int)(gridDim.z * blockDim.x)) *
    gridDim.z * blockDim.x;

  for (int i = C_I_Tile * blockIdx.x; i < ubi; i += C_I_Tile * gridDim.x) {
    for (int j = C_J_Tile * (blockIdx.y * blockDim.y + threadIdx.y);
         j < ubj;
         j += C_J_Tile * gridDim.y * blockDim.y) {
      for (int xy = blockDim.x * blockIdx.z + threadIdx.x;
           xy < ubxy;
           xy += gridDim.z * blockDim.x) {
        Complex a[C_I_Tile];
        Complex b[C_J_Tile][ReductionUnroll];
        Complex c[C_I_Tile][C_J_Tile];

        for (int k = 0; k < ubk; k += ReductionUnroll) {
          // Kill WAW dependence
          __syncthreads();

          // Load B from device to registers with boundary check and static
          // optimization of those checks.
          for (int jj = 0; jj < C_J_Tile; ++jj) {
            if ((StaticUnrollXY || xy < NumBatches) &&
                (StaticUnrollB ||
                 (ConjugateTransposeB && j + jj < B.getSize(0)) ||
                 (!ConjugateTransposeB && j + jj < B.getSize(1)))) {
              for (int kk = 0; kk < ReductionUnroll; ++kk) {
                b[jj][kk] = (StaticUnrollReduction || k + kk < numRed) ?
                  ((ConjugateTransposeB) ?
                    ldg(&B[j + jj][k + kk][xy]).conjugate() :
                    ldg(&B[k + kk][j + jj][xy]))
                  : Complex(0.0f);
              }
            } else {
              for (int kk = 0; kk < ReductionUnroll; ++kk) {
                b[jj][kk] = Complex(0.0f);
              }
            }
          }

          // Load A from device to shared with boundary check and static
          // optimization of those checks.
          // Distribute loads across blockIdx.y
          __shared__ Complex
            as[C_I_Tile][ReductionUnroll][C_XY_Placement_ThreadIdx_X + 1];
          assert(C_I_Tile <= blockDim.y);
          if (threadIdx.y < C_I_Tile) {
            int ii = threadIdx.y;
            if ((StaticUnrollXY || xy < NumBatches) &&
                (StaticUnrollA ||
                 (ConjugateTransposeA && i + ii < A.getSize(1)) ||
                 (!ConjugateTransposeA && i + ii < A.getSize(0)))) {
              for (int kk = 0; kk < ReductionUnroll; ++kk) {
                as[ii][kk][threadIdx.x] =
                  (StaticUnrollReduction || k + kk < numRed) ?
                    ((ConjugateTransposeA) ?
                     ldg(&A[k + kk][i + ii][xy]).conjugate() :
                     ldg(&A[i + ii][k + kk][xy]))
                  : Complex(0.0f);
              }
            } else {
              for (int kk = 0; kk < ReductionUnroll; ++kk) {
                as[ii][kk][threadIdx.x] = Complex(0.0f);
              }
            }
          }

          // Use init to hide some latencies
          if (k == 0) {
            for (int ii = 0; ii < C_I_Tile; ++ii) {
              for (int jj = 0; jj < C_J_Tile; ++jj) {
                c[ii][jj] =
                  (Accumulate &&
                   (StaticUnrollCI || i + ii < C.getSize(0)) &&
                   (StaticUnrollCJ || j + jj < C.getSize(1)) &&
                   (StaticUnrollXY || xy < NumBatches)) ?
                     C[i + ii][j + jj][xy] : Complex(0.0f);
              }
            }
          }

          // Kill RAW dependence
          __syncthreads();

          // Perform partial accumulation
          for (int kk = 0; kk < ReductionUnroll; ++kk) {
            for (int ii = 0; ii < C_I_Tile; ++ii) {
              a[ii] = as[ii][kk][threadIdx.x];
            }
            for (int jj = 0; jj < C_J_Tile; ++jj) {
              for (int ii = 0; ii < C_I_Tile; ++ii) {
                c[ii][jj] = a[ii] * b[jj][kk] + c[ii][jj];
              }
            }
          }
        }

        if (StaticUnrollXY || xy < NumBatches) {
          for (int ii = 0;
               ii < C_I_Tile && (StaticUnrollCI || i + ii < C.getSize(0));
               ++ii) {
            for (int jj = 0;
                 jj < C_J_Tile && (StaticUnrollCJ || j + jj < C.getSize(1));
                 ++jj) {
              c[ii][jj].re() *= invNorm.re();
              c[ii][jj].im() *= invNorm.re();
              C[i + ii][j + jj][xy] = c[ii][jj];
            }
          }
        }
      }
    }
  }
}

struct HalfFtor {
  HalfFtor() {}
  void operator()(int& n) { n >>= 1; }
};

} // ns detail

template
<int Dim, bool ConjugateTransposeA, bool ConjugateTransposeB, bool Accumulate>
void transposeMM(DeviceTensor<float, Dim>& A,
                 DeviceTensor<float, Dim>& B,
                 DeviceTensor<float, Dim>& C,
                 float invNorm,
                 cudaStream_t s) {
  int szA[Dim - 1];
  int stA[Dim - 1];
  std::copy(A.sizes(), A.sizes() + Dim - 1, szA);
  std::copy(A.strides(), A.strides() + Dim - 1, stA);
  std::for_each(&stA[0], &stA[Dim - 1], detail::HalfFtor());

  int szB[Dim - 1];
  int stB[Dim - 1];
  std::copy(B.sizes(), B.sizes() + Dim - 1, szB);
  std::copy(B.strides(), B.strides() + Dim - 1, stB);
  std::for_each(&stB[0], &stB[Dim - 1], detail::HalfFtor());

  int szC[Dim - 1];
  int stC[Dim - 1];
  std::copy(C.sizes(), C.sizes() + Dim - 1, szC);
  std::copy(C.strides(), C.strides() + Dim - 1, stC);
  std::for_each(&stC[0], &stC[Dim - 1], detail::HalfFtor());

  DeviceTensor<Complex, Dim - 1> cA(A.template dataAs<Complex>(), szA, stA);
  DeviceTensor<Complex, Dim - 1> cB(B.template dataAs<Complex>(), szB, stB);
  DeviceTensor<Complex, Dim - 1> cC(C.template dataAs<Complex>(), szC, stC);

  DeviceTensor<Complex, 3> dcA = cA.template downcastInner<3>();
  DeviceTensor<Complex, 3> dcB = cB.template downcastInner<3>();
  DeviceTensor<Complex, 3> dcC = cC.template downcastInner<3>();

#define INSTANTIATE_FBMM_FULLY_UNROLLED(                                \
  C_J_Unroll,                                                           \
  C_XY_Placement_ThreadIdx_X,                                           \
  C_XY_Placement_BlockIdx_Z,                                            \
  C_I_Tile,                                                             \
  C_J_Tile,                                                             \
  ReductionUnroll)                                                      \
  {                                                                     \
    bool StaticUnrollA =                                                \
      ((ConjugateTransposeA && (dcA.getSize(1) % C_I_Tile == 0)) ||     \
       (!ConjugateTransposeA && (dcA.getSize(0) % C_I_Tile == 0)));     \
    bool StaticUnrollB =                                                \
      ((ConjugateTransposeB && (dcB.getSize(0) % C_J_Tile == 0)) ||     \
       (!ConjugateTransposeB && (dcB.getSize(1) % C_J_Tile == 0)));     \
    bool StaticUnrollCI = (dcC.getSize(0) % C_I_Tile == 0);             \
    bool StaticUnrollCJ = (dcC.getSize(1) % (C_J_Unroll * C_J_Tile) == 0); \
    bool StaticUnrollXY =                                               \
      (dcC.getSize(2) %                                                 \
       (C_XY_Placement_BlockIdx_Z * C_XY_Placement_ThreadIdx_X) == 0);  \
    const int numRed =                                                  \
      (ConjugateTransposeA) ? dcA.getSize(0) : dcA.getSize(1);          \
    bool StaticUnrollReduction = (numRed % ReductionUnroll == 0);       \
    count++;                                                            \
    if (debug) {                                                        \
      LOG(INFO) << StaticUnrollA << " " << StaticUnrollB << " "         \
                << StaticUnrollCI << " " << StaticUnrollCJ << " "       \
                << StaticUnrollXY << " " << StaticUnrollReduction;      \
    }                                                                   \
    if (StaticUnrollA && StaticUnrollB && StaticUnrollCI &&             \
        StaticUnrollCJ && StaticUnrollXY && StaticUnrollReduction) {    \
      if (debug) {                                                      \
        LOG(INFO) << "Count: " << count;                                \
      }                                                                 \
      /* Needed for proper loading of data */                           \
      CHECK_LE(C_I_Tile, C_J_Unroll);                                   \
      dim3 blocks(ceil(dcC.getSize(0), C_I_Tile),                       \
                  ceil(dcC.getSize(1), C_J_Unroll * C_J_Tile),          \
                  C_XY_Placement_BlockIdx_Z                             \
                 );                                                     \
      dim3 threads(C_XY_Placement_ThreadIdx_X, C_J_Unroll);             \
      detail::transposeMMTiledKernel<ConjugateTransposeA,               \
                                     ConjugateTransposeB,               \
                                     C_XY_Placement_ThreadIdx_X,        \
                                     C_J_Unroll,                        \
                                     C_I_Tile,                          \
                                     C_J_Tile,                          \
                                     ReductionUnroll,                   \
                                     true,                              \
                                     true,                              \
                                     true,                              \
                                     true,                              \
                                     true,                              \
                                     true,                              \
                                     Accumulate>                        \
        <<<blocks, threads, 0, s>>>(dcA, dcB, dcC, Complex(invNorm));   \
      return;                                                           \
    }                                                                   \
  }

  int count = 0;
  bool debug = false;
  if (debug) {
    LOG(INFO) << "ConjugateTransposeA: " << ConjugateTransposeA
              << " ConjugateTransposeB: " << ConjugateTransposeB
              << "\nA: " << A << " -> " << cA << " -> " << dcA
              << "\nB: " << B << " -> " << cB << " -> " << dcB
              << "\nC: " << C << " -> " << cC << " -> " << dcC;
  }

  // INSTANTIATE_FBMM_FULLY_UNROLLED(C_J_Unroll,
  //                                 C_XY_Placement_ThreadIdx_X,
  //                                 C_XY_Placement_BlockIdx_Z,
  //                                 C_I_Tile,
  //                                 C_J_Tile,
  //                                 ReductionUnroll)
  // TODO: Add more instantiations to cover use cases properly
  if (dcA.getSize(2) == 3 * 4) {
  } else if (dcA.getSize(2) == 5 * 8) {
  } else if (dcA.getSize(2) == 9 * 16) {
    // Imagenet 4-GPU model parallel 128x256x96
    INSTANTIATE_FBMM_FULLY_UNROLLED(128, 8, 6, /* */ 8, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(96,  8, 6, /* */ 8, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(64, 12, 6, /* */ 8, 2, 2);
    // Imagenet 4-GPU model parallel 128x96x96
    INSTANTIATE_FBMM_FULLY_UNROLLED(16, 12, 6, /* */ 8, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(12, 12, 6, /* */ 8, 2, 2);
    // TODO: VGG first layer, pretty bad use case for this fbmm because
    // of the 3 dimension in 64 x 3 x 64

    /////////////////////////////////////////////////////////////
    // Duplicate for unroll dimension of size 1 (batch size == 1)
    /////////////////////////////////////////////////////////////

    // Imagenet 4-GPU model parallel 128x256x96
    INSTANTIATE_FBMM_FULLY_UNROLLED(128, 8, 6, /* */ 8, 2, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(96,  8, 6, /* */ 8, 2, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(64, 12, 6, /* */ 8, 2, 1);
    // Imagenet 4-GPU model parallel 128x96x96
    INSTANTIATE_FBMM_FULLY_UNROLLED(16, 12, 6, /* */ 8, 2, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(12, 12, 6, /* */ 8, 2, 1);
    // TODO: VGG first layer, pretty bad use case for this fbmm because
    // of the 3 dimension in 64 x 3 x 64
  } else if (dcA.getSize(2) == 17 * 32) {
    // Imagenet 4-GPU model parallel 128x24x64
    INSTANTIATE_FBMM_FULLY_UNROLLED(8, 16, 17, /* */ 8, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(4, 16, 17, /* */ 4, 2, 2);
    // VGG first layer, pretty bad use case for this fbmm because
    // of the 3 dimension in 64 x 3 x 64
    INSTANTIATE_FBMM_FULLY_UNROLLED(8, 16, 17, /* */ 3, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(4, 16, 17, /* */ 3, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(8, 16, 17, /* */ 2, 3, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(4, 16, 17, /* */ 2, 3, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(8, 16, 17, /* */ 2, 2, 3);
    INSTANTIATE_FBMM_FULLY_UNROLLED(4, 16, 17, /* */ 2, 2, 3);
    INSTANTIATE_FBMM_FULLY_UNROLLED(1, 16, 17, /* */ 1, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(1, 16, 17, /* */ 1, 3, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(1, 16, 17, /* */ 1, 2, 3);

    /////////////////////////////////////////////////////////////
    // Duplicate for unroll dimension of size 1 (batch size == 1)
    /////////////////////////////////////////////////////////////

    // Imagenet 4-GPU model parallel 128x24x64
    INSTANTIATE_FBMM_FULLY_UNROLLED(8, 16, 17, /* */ 8, 2, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(4, 16, 17, /* */ 4, 2, 1);
    // VGG first layer, pretty bad use case for this fbmm because
    // of the 3 dimension in 64 x 3 x 64
    INSTANTIATE_FBMM_FULLY_UNROLLED(8, 16, 17, /* */ 3, 2, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(4, 16, 17, /* */ 3, 2, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(8, 16, 17, /* */ 2, 3, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(4, 16, 17, /* */ 2, 3, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(8, 16, 17, /* */ 2, 2, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(4, 16, 17, /* */ 2, 2, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(1, 16, 17, /* */ 1, 2, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(1, 16, 17, /* */ 1, 3, 1);
    INSTANTIATE_FBMM_FULLY_UNROLLED(1, 16, 17, /* */ 1, 2, 1);
  } else if (dcA.getSize(2) == 33 * 64) {
  } else if (dcA.getSize(2) == 65 * 128) {
  }

  // Fallback cases
  if (debug) {
    LOG(WARNING) << "Fallback case, performance will be bad";
  }
  INSTANTIATE_FBMM_FULLY_UNROLLED(8, 8, 4, /* */ 8, 2, 2);
  INSTANTIATE_FBMM_FULLY_UNROLLED(4, 8, 2, /* */ 4, 2, 2);
  INSTANTIATE_FBMM_FULLY_UNROLLED(4, 4, 1, /* */ 4, 2, 2);
  INSTANTIATE_FBMM_FULLY_UNROLLED(8, 8, 4, /* */ 8, 2, 1);
  INSTANTIATE_FBMM_FULLY_UNROLLED(4, 8, 2, /* */ 4, 2, 1);
  INSTANTIATE_FBMM_FULLY_UNROLLED(4, 4, 1, /* */ 4, 2, 1);

  if (debug) {
    LOG(WARNING) << "Unspecialized case, performance will be very bad";
  }

  // Default case, performance wil most likely be bad if we get here
#define C_I_Tile 8
#define C_J_Tile 2
#define ReductionUnroll 1
#define C_J_Unroll 16
#define C_XY_Placement_ThreadIdx_X 12
#define C_XY_Placement_BlockIdx_Z 6

  dim3 blocks(ceil(dcC.getSize(0), C_I_Tile),
              ceil(dcC.getSize(1), C_J_Unroll * C_J_Tile),
              C_XY_Placement_BlockIdx_Z);
  dim3 threads(C_XY_Placement_ThreadIdx_X, C_J_Unroll);
  detail::transposeMMTiledKernel<ConjugateTransposeA,
                                 ConjugateTransposeB,
                                 C_XY_Placement_ThreadIdx_X,
                                 C_J_Unroll,
                                 C_I_Tile,
                                 C_J_Tile,
                                 ReductionUnroll,
                                 false,
                                 false,
                                 false,
                                 false,
                                 false,
                                 false,
                                 Accumulate>
    <<<blocks, threads, 0, s>>>(dcA, dcB, dcC, Complex(invNorm));
}

}} // ns
