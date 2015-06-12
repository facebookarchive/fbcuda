// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

namespace facebook { namespace cuda { namespace fbfft {

class FBFFTParameters {
 public:
  enum ErrorCode {
    Success = 0,
    UnsupportedSize = 1,
    UnsupportedDimension = 2,
    CudaError = 3
  };

  FBFFTParameters() :
      direction_(true),
      normalize_(false),
      padLeft_(0),
      padUp_(0)
    {}

  FBFFTParameters& forward() {
    direction_ = true;
    return *this;
  }

  FBFFTParameters& inverse() {
    direction_ = false;
    return *this;
  }

  FBFFTParameters& normalize(bool n) {
    normalize_ = n;
    return *this;
  }

  FBFFTParameters& withPadLeft(int p) {
    padLeft_ = p;
    return *this;
  }

  FBFFTParameters& withPadUp(int p) {
    padUp_ = p;
    return *this;
  }

  bool forwardFFT() const { return  direction_; }
  bool inverseFFT() const { return !direction_; }
  bool normalizeFFT() const { return normalize_; }
  int padLeft() const { return padLeft_; }
  int padUp() const { return padUp_; }

 private:
  bool direction_;
  bool normalize_;
  int padLeft_;
  int padUp_;
};

}}}
