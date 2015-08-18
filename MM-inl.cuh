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

__device__ __forceinline__ constexpr int max(int i, int j) {
  return (i < j) ? j : i;
}

__device__ __forceinline__ constexpr int max(int i, int j, int k) {
  return facebook::cuda::detail::max(facebook::cuda::detail::max(i, j), k);
}

__device__ __forceinline__ Complex ldg(const Complex* p) {
  return Complex(__ldg((const float2*)p));
}

__device__ __forceinline__ void ldg(Complex& c1, Complex&c2, const Complex* p) {
  const float4 f = __ldg((const float4*)p);
  c1 = Complex(f.x, f.y);
  c2 = Complex(f.z, f.w);
}

template <bool ConjugateTransposeA,
          bool ConjugateTransposeB,
          int FFTSize,
          int FFTElements,
          int TileI,
          int TileJ,
          int TileK,
          int TileIThreadIdxY,
          int TileJThreadIdxZ,
          bool Accumulate>
__launch_bounds__(32 * 4 * 2, 2) // 128 registers on K40
__global__ void transposeMMTiledKernelSmall(const DeviceTensor<Complex, 3> A,
                                            const DeviceTensor<Complex, 3> B,
                                            DeviceTensor<Complex, 3> C,
                                            Complex invNorm)
{
  const auto xyBase = blockIdx.z * blockDim.x;
  const auto xy = blockIdx.z * blockDim.x + threadIdx.x;
  const int numRed = (ConjugateTransposeA) ? A.getSize(0) : A.getSize(1);

  // Conditions must hold for float4 implementation to be valid
  assert(xy < FFTSize * (FFTSize / 2 + 1));
  assert(FFTElements == blockDim.x);
  assert(TileIThreadIdxY == blockDim.y);
  assert(TileJThreadIdxZ == blockDim.z);
  assert(numRed % TileK == 0);

  Complex c[TileI][TileJ];

  // for (int i = TileI * blockIdx.x; i < C.getSize(0); i += TileI * gridDim.x) {
  //   for (int j = TileJ * blockIdx.y; j < C.getSize(1); j += TileJ * gridDim.y) {
  {
    {
      // blockIdx.x/y are the ceils
      int i = TileI * (threadIdx.y + blockDim.y * blockIdx.x);
      int j = TileJ * (threadIdx.z + blockDim.z * blockIdx.y);

      // Guard against overflows
      assert(i + TileI <= C.getSize(0));
      assert(j + TileJ <= C.getSize(1));

      for (int ii = 0; ii < TileI; ++ii) {
        for (int jj = 0; jj < TileJ; ++jj) {
          c[ii][jj] = (Accumulate) ?
            C[i + ii][j + jj][xy] : Complex(0.0f);
        }
      }

      for (int k = 0; k < numRed; k += TileK) {
        Complex a[TileK][TileI];
        Complex b[TileK][TileJ];

        __shared__ Complex swap
          [TileJThreadIdxZ]
          [TileIThreadIdxY]
          [facebook::cuda::detail::max(TileI, TileJ, TileK)]
          [2]
          [FFTElements];
        // View float2[2][FFTElements] as float4[FFTElements], let the
        // compiler worry about the indexing.
        auto swapViewFloat4 =
          (float4(*)
           [TileIThreadIdxY]
           [facebook::cuda::detail::max(TileI, TileJ, TileK)]
           [FFTElements])swap;

        // Illustration with blockDim.x == 8
        // Goal
        // th  0  1  2  3  4  5  6  7
        // a  A0 A1 A2 A3 A4 A5 A6 A7
        // b  B0 B1 B2 B3 B4 B5 B6 B7
        //
        // Threads  < blockDim.x / 2 load A0 - A7 into shared float4
        // Threads >= blockDim.x / 2 load B0 - B7 into shared float4
        // Actual
        // s  A0/A1 A2/A3 A4/A5 A6/A7 | B0/B1 B2/B3 B4/B5 B6/B7
        const auto xdim = (threadIdx.x < blockDim.x / 2) ?
          xyBase + 2 * threadIdx.x :
          xyBase + 2 * (threadIdx.x - blockDim.x / 2);
        for (int kk = 0; kk < TileK; ++kk) {
          // This statically unrolls for max(TileI, TileJ, TileK) and computes
          // a base pointer for Threads < blockDim.x / 2 and
          // Threads >= blockDim.x / 2
          // If there is imbalance, the pointer computed is nullptr
          // and the load is not generated.
          for (int ij = 0;
               ij < facebook::cuda::detail::max(TileI, TileJ, TileK); ++ij) {
            const Complex* baseA = (ij >= TileI) ?
              nullptr :
              ((!ConjugateTransposeA) ?
               A[i + ij][k + kk][xdim].data() :
               A[k + kk][i + ij][xdim].data()) ;

            const Complex* baseB = (ij >= TileJ) ?
              nullptr :
              ((!ConjugateTransposeB) ?
               B[k + kk][j + ij][xdim].data() :
               B[j + ij][k + kk][xdim].data()) ;

            const Complex* base =
              (threadIdx.x < blockDim.x / 2) ? baseA : baseB;

            if (base) {
              swapViewFloat4[threadIdx.z][threadIdx.y][ij][threadIdx.x] =
                __ldg((const float4*)(base));
            }
           }

          for (int ii = 0; ii < TileI; ++ii) {
            a[kk][ii] = swap[threadIdx.z][threadIdx.y][ii][0][threadIdx.x];
          }
          for (int jj = 0; jj < TileJ; ++jj) {
            b[kk][jj] = swap[threadIdx.z][threadIdx.y][jj][1][threadIdx.x];
           }
        }

        if (ConjugateTransposeA) {
          for (int kk = 0; kk < TileK; ++kk) {
            for (int ii = 0; ii < TileI; ++ii) {
              a[kk][ii] = a[kk][ii].conjugate();
            }
          }
        }
        if (ConjugateTransposeB) {
          for (int kk = 0; kk < TileK; ++kk) {
            for (int jj = 0; jj < TileJ; ++jj) {
              b[kk][jj] = b[kk][jj].conjugate();
            }
          }
        }

        for (int kk = 0; kk < TileK; ++kk) {
          for (int jj = 0; jj < TileJ; ++jj) {
            for (int ii = 0; ii < TileI; ++ii) {
              c[ii][jj] += a[kk][ii] * b[kk][jj];
             }
           }
         }
       }

      // Actual
      // c  C0 C2 C4 C6 C1 C3 C5 C7
      for (int ii = 0; ii < TileI; ++ii) {
        for (int jj = 0; jj < TileJ; ++jj) {
          c[ii][jj].re() *= invNorm.re();
          c[ii][jj].im() *= invNorm.re();
          *(C[i + ii][j + jj][xy].dataAs<float2>()) = (float2)(c[ii][jj]);
        }
      }
     }
   }
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
              *(C[i + ii][j + jj][xy].dataAs<float2>()) = (float2)(c[ii][jj]);
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
    if (debug) {                                                        \
      LOG(INFO) << StaticUnrollA << " " << StaticUnrollB << " "         \
                << StaticUnrollCI << " " << StaticUnrollCJ << " "       \
                << StaticUnrollXY << " " << StaticUnrollReduction;      \
      LOG(INFO) << StaticUnrollA << " " << StaticUnrollB << " "         \
                << StaticUnrollCI << " " << StaticUnrollCJ << " "       \
                << StaticUnrollXY << " " << StaticUnrollReduction;      \
    }                                                                   \
    if (StaticUnrollA && StaticUnrollB && StaticUnrollCI &&             \
        StaticUnrollCJ && StaticUnrollXY && StaticUnrollReduction) {    \
      if (debug) {                                                      \
        LOG(INFO) << "Params: " << C_J_Unroll << " " <<                 \
          C_XY_Placement_ThreadIdx_X << " " <<                          \
          C_XY_Placement_BlockIdx_Z << " " <<                           \
          C_I_Tile << " " <<                                            \
          C_J_Tile << " " <<                                            \
          ReductionUnroll;                                              \
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


#define INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                     \
  TILEI, TILEJ, TILEK, TILEITHY, TILEJTHZ, FFTELEMENTS)                 \
  {                                                                     \
    constexpr int TileI = TILEI;                                        \
    constexpr int TileJ = TILEJ;                                        \
    constexpr int TileK = TILEK;                                        \
    constexpr int TileIThreadIdxY = TILEITHY;                           \
    constexpr int TileJThreadIdxZ = TILEJTHZ;                           \
    constexpr int FFTElements = FFTELEMENTS;                            \
    if (dcC.getSize(0) % (TileI * TileIThreadIdxY) == 0 &&              \
        dcC.getSize(1) % (TileJ * TileJThreadIdxZ) == 0 &&              \
        (                                                               \
          (!ConjugateTransposeA && ((dcA.getSize(1) % TileK) == 0)) ||  \
          ( ConjugateTransposeA && ((dcA.getSize(0) % TileK) == 0))     \
        )                                                          &&   \
        (FFTSize * (FFTSize / 2 + 1)) % (2 * FFTElements) == 0) {       \
      if (debug) {                                                      \
        LOG(INFO) << " TileI = " << TileI                               \
                  << " TileJ = " << TileJ                               \
                  << " TileK = " << TileK                               \
                  << " TileIThreadIdxY = " << TileIThreadIdxY           \
                  << " TileJThreadIdxZ = " << TileJThreadIdxZ           \
                  << " FFTElements = " << FFTElements;                  \
      }                                                                 \
      static_assert(FFTSize % FFTElements == 0,                         \
                    "float4 reads requires FFTSize % FFTElements == 0");\
      dim3 blocks(ceil(dcC.getSize(0), TileI * TileIThreadIdxY),        \
                  ceil(dcC.getSize(1), TileJ * TileJThreadIdxZ),        \
                  ceil(FFTSize * (FFTSize / 2 + 1), FFTElements));      \
      dim3 threads(FFTElements, TileIThreadIdxY, TileJThreadIdxZ);      \
      detail::transposeMMTiledKernelSmall<ConjugateTransposeA,          \
                                          ConjugateTransposeB,          \
                                          FFTSize,                      \
                                          FFTElements,                  \
                                          TileI,                        \
                                          TileJ,                        \
                                          TileK,                        \
                                          TileIThreadIdxY,              \
                                          TileJThreadIdxZ,              \
                                          Accumulate>                   \
        <<<blocks, threads, 0, s>>> (dcA, dcB, dcC, Complex(invNorm));  \
      return;                                                           \
    }                                                                   \
  }

// Always look permutations of (TILEI, TILEJ, TILEK) and (TILEITHY, TILEJTHZ)
#define INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(                          \
  TILEI, TILEJ, TILEK, TILEITHY, TILEJTHZ, FFTELEMENTS)                 \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEI, TILEJ, TILEK, TILEITHY, TILEJTHZ, FFTELEMENTS);              \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEI, TILEJ, TILEK, TILEJTHZ, TILEITHY, FFTELEMENTS);              \
                                                                        \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEI, TILEK, TILEJ, TILEITHY, TILEJTHZ, FFTELEMENTS);              \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEI, TILEK, TILEJ, TILEJTHZ, TILEITHY, FFTELEMENTS);              \
                                                                        \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEJ, TILEI, TILEK, TILEITHY, TILEJTHZ, FFTELEMENTS);              \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEJ, TILEI, TILEK, TILEJTHZ, TILEITHY, FFTELEMENTS);              \
                                                                        \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEJ, TILEK, TILEI, TILEITHY, TILEJTHZ, FFTELEMENTS);              \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEJ, TILEK, TILEI, TILEJTHZ, TILEITHY, FFTELEMENTS);              \
                                                                        \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEK, TILEI, TILEJ, TILEITHY, TILEJTHZ, FFTELEMENTS);              \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEK, TILEI, TILEJ, TILEJTHZ, TILEITHY, FFTELEMENTS);              \
                                                                        \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEK, TILEJ, TILEI, TILEITHY, TILEJTHZ, FFTELEMENTS);              \
  INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED_IMPL(                           \
    TILEK, TILEJ, TILEI, TILEJTHZ, TILEITHY, FFTELEMENTS);

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
    INSTANTIATE_FBMM_FULLY_UNROLLED(32,  6, 5, /* */ 8, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(16,  8, 5, /* */ 8, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(12,  8, 5, /* */ 8, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(8,  8, 5, /* */ 8, 2, 2);

    constexpr int FFTSize = 8;
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 4, 4, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 4, 2, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 4, 1, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 2, 2, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 2, 1, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 1, 1, 4);

    // InputPlane = 3*k (RGB input mostly)
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 4, 4, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 4, 2, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 4, 1, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 2, 2, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 2, 1, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 1, 1, 4);

    // Batch size = 1
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 4, 4, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 4, 2, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 4, 1, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 2, 2, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 2, 1, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 1, 1, 4);

    // Fallback
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(2, 2, 2, 1, 1, 4);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 1, 1, 1, 1, 4);
  } else if (dcA.getSize(2) == 9 * 16) {
    INSTANTIATE_FBMM_FULLY_UNROLLED(16, 12, 6, /* */ 8, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(12, 12, 6, /* */ 8, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(8, 12, 6, /* */ 8, 2, 2);

    constexpr int FFTSize = 16;
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 4, 2, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 4, 1, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 2, 2, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 2, 1, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 1, 1, 8);

    // InputPlane = 3*k (RGB input mostly)
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 4, 2, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 4, 1, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 2, 2, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 2, 1, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 1, 1, 8);

    // Batch size = 1
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 4, 2, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 4, 1, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 2, 2, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 2, 1, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 1, 1, 8);

    // Fallback
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(2, 2, 2, 1, 1, 8);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 1, 1, 1, 1, 8);
  } else if (dcA.getSize(2) == 17 * 32) {
    INSTANTIATE_FBMM_FULLY_UNROLLED(8, 16, 17, /* */ 8, 2, 2);
    INSTANTIATE_FBMM_FULLY_UNROLLED(4, 16, 17, /* */ 4, 2, 2);

    constexpr int FFTSize = 32;
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 2, 2, 16);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(4, 4, 4, 1, 1, 16);

    // InputPlane = 3*k (RGB input mostly)
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 1, 4, 16);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 1, 2, 16);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(3, 4, 4, 1, 1, 16);

    // Batch size = 1
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 1, 4, 16);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 1, 2, 16);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 4, 4, 1, 1, 16);

    // Fallback
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(2, 2, 2, 1, 1, 16);
    INSTANTIATE_FBMM_SMALL_FULLY_UNROLLED(1, 1, 1, 1, 1, 16);
  } else if (dcA.getSize(2) == 33 * 64) {
  } else if (dcA.getSize(2) == 65 * 128) {
  }

  // Fallback cases
  if (debug) {
    LOG(WARNING) << "Unspecialized case, performance will be very bad";
  }

  // Default case, performance wil most likely be bad if we get here
#define C_I_Tile 4
#define C_J_Tile 2
#define ReductionUnroll 1
#define C_J_Unroll 4
#define C_XY_Placement_ThreadIdx_X 4
#define C_XY_Placement_BlockIdx_Z 1

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
