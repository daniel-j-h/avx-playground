#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <immintrin.h>
#include "VecBase.h"

namespace avx {

using vec8f = vec<float, 8u>;

template <>
struct vec<float, 8u> final {
  using Value = float;
  static const constexpr std::size_t Size = 8u;

  // thin abstraction; instead of explicit conversion operator and the need for static-casting, just use .ymm
  __m256 ymm;

  vec() : ymm(_mm256_undefined_ps()) {}
  vec(__m256 x) : ymm(x) {}
  vec(Value scalar) { load(scalar); }
  vec(const Value* first, const Value* last) { load(first, last); }
  vec(Value e7, Value e6, Value e5, Value e4, Value e3, Value e2, Value e1, Value e0) {
    load(e7, e6, e5, e4, e3, e2, e1, e0);
  }

  // load
  void load(Value scalar) { ymm = _mm256_set1_ps(scalar); }

  void load(const Value* first, const Value* last) {
    assert(last - first == Size);
    ymm = _mm256_loadu_ps(first);
  }

  void load_aligned(const Value* first, const Value* last) {
    assert(last - first == Size);
    ymm = _mm256_load_ps(first);
  }

  void load(Value e7, Value e6, Value e5, Value e4, Value e3, Value e2, Value e1, Value e0) {
    ymm = _mm256_setr_ps(e7, e6, e5, e4, e3, e2, e1, e0);
  }

  // store
  void store(Value* first, Value* last) {
    assert(last - first == Size);
    _mm256_storeu_ps(first, ymm);
  }

  void store_aligned(Value* first, Value* last) {
    assert(last - first == Size);
    _mm256_store_ps(first, ymm);
  }

  void store_aligned_stream(Value* first, Value* last) {
    assert(last - first == Size);
    _mm256_stream_ps(first, ymm);
  }

  // misc
  void zero() { ymm = _mm256_setzero_ps(); }

  // compound ops, no implicit conversion needed for lhs
  vec8f& operator+=(const vec8f& rhs) {
    ymm = _mm256_add_ps(ymm, rhs.ymm);
    return *this;
  }

  vec8f& operator-=(const vec8f& rhs) {
    ymm = _mm256_sub_ps(ymm, rhs.ymm);
    return *this;
  }

  vec8f& operator*=(const vec8f& rhs) {
    ymm = _mm256_mul_ps(ymm, rhs.ymm);
    return *this;
  }

  vec8f& operator/=(const vec8f& rhs) {
    ymm = _mm256_div_ps(ymm, rhs.ymm);
    return *this;
  }

  vec8f& operator|=(const vec8f& rhs) {
    ymm = _mm256_or_ps(ymm, rhs.ymm);
    return *this;
  }

  vec8f& operator&=(const vec8f& rhs) {
    ymm = _mm256_and_ps(ymm, rhs.ymm);
    return *this;
  }

  vec8f& operator^=(const vec8f& rhs) {
    ymm = _mm256_xor_ps(ymm, rhs.ymm);
    return *this;
  }

  // subscript
  Value& operator[](std::size_t index) {
    assert(index < Size);
    return ymm[index];
  }

  const Value& operator[](std::size_t index) const {
    assert(index < Size);
    return ymm[index];
  }
};


// free standing functions, participate in implicit conversion for lhs and rhs
inline vec8f operator+(const vec8f& lhs, const vec8f& rhs) { return {_mm256_add_ps(lhs.ymm, rhs.ymm)}; }
inline vec8f operator-(const vec8f& lhs, const vec8f& rhs) { return {_mm256_sub_ps(lhs.ymm, rhs.ymm)}; }
inline vec8f operator*(const vec8f& lhs, const vec8f& rhs) { return {_mm256_mul_ps(lhs.ymm, rhs.ymm)}; }
inline vec8f operator/(const vec8f& lhs, const vec8f& rhs) { return {_mm256_div_ps(lhs.ymm, rhs.ymm)}; }

// TODO(daniel): andnot, op%, op~, op!, <<, >>, ++, --
inline vec8f operator|(const vec8f& lhs, const vec8f& rhs) { return {_mm256_or_ps(lhs.ymm, rhs.ymm)}; }
inline vec8f operator&(const vec8f& lhs, const vec8f& rhs) { return {_mm256_and_ps(lhs.ymm, rhs.ymm)}; }
inline vec8f operator^(const vec8f& lhs, const vec8f& rhs) { return {_mm256_xor_ps(lhs.ymm, rhs.ymm)}; }

// unary
inline vec8f operator~(const vec8f& x) { return {_mm256_xor_ps(x.ymm, _mm256_set1_ps(-1.f))}; }
inline vec8f operator-(const vec8f& x) { return {_mm256_sub_ps(_mm256_set1_ps(0.f), x.ymm)}; }


// O = ordered, U = unordered, S = signaling, Q = non-signaling
// CMP_EQ_OQ = 0
// CMP_NEQ_UQ = 4
// CMP_LT_OS = 1
// CMP_LE_OS = 2
// CMP_GE_OS = 13
// CMP_GT_OS = 14
template <int ComparisonMask8Bit>
inline vec8f compare(const vec8f& lhs, const vec8f& rhs) { return _mm256_cmp_ps(lhs.ymm, rhs.ymm, ComparisonMask8Bit); }

inline vec8f operator==(const vec8f& lhs, const vec8f& rhs) { return compare<0>(lhs, rhs); }
inline vec8f operator!=(const vec8f& lhs, const vec8f& rhs) { return compare<4>(lhs, rhs); }
inline vec8f operator<(const vec8f& lhs, const vec8f& rhs) { return compare<1>(lhs, rhs); }
inline vec8f operator>(const vec8f& lhs, const vec8f& rhs) { return compare<14>(lhs, rhs); }
inline vec8f operator<=(const vec8f& lhs, const vec8f& rhs) { return compare<2>(lhs, rhs); }
inline vec8f operator>=(const vec8f& lhs, const vec8f& rhs) { return compare<13>(lhs, rhs); }


// misc
template <int BlendMask8Bit>
inline vec8f blend(const vec8f& lhs, const vec8f& rhs) { return {_mm256_blend_ps(lhs.ymm, rhs.ymm, BlendMask8Bit)}; }
inline vec8f blend(const vec8f& lhs, const vec8f& rhs, const vec8f& mask) { return {_mm256_blendv_ps(lhs.ymm, rhs.ymm, mask.ymm)}; }

template <int PermuteMask8Bit>
inline vec8f permute(const vec8f& x) { return {_mm256_permute_ps(x.ymm, PermuteMask8Bit)}; }

template <int PermuteMask8Bit>
inline vec8f permute(const vec8f& lhs, const vec8f& rhs) { return {_mm256_permute2f128_ps(lhs.ymm, rhs.ymm, PermuteMask8Bit)}; }

template <int ShuffleMask8Bit>
inline vec8f shuffle(const vec8f& lhs, const vec8f& rhs) { return {_mm256_shuffle_ps(lhs.ymm, rhs.ymm, ShuffleMask8Bit)}; }

inline vec8f unpackHigh(const vec8f& lhs, const vec8f& rhs) { return {_mm256_unpackhi_ps(lhs.ymm, rhs.ymm)}; }
inline vec8f unpackLow(const vec8f& lhs, const vec8f& rhs) { return {_mm256_unpacklo_ps(lhs.ymm, rhs.ymm)}; }


// math
inline vec8f ceil(const vec8f& x) { return {_mm256_ceil_ps(x.ymm)}; }
inline vec8f floor(const vec8f& x) { return {_mm256_floor_ps(x.ymm)}; }

template <int RoundingMode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC>
inline vec8f round(const vec8f& x) { return {_mm256_round_ps(x.ymm, RoundingMode)}; }

inline vec8f min(const vec8f& lhs, vec8f& rhs) { return {_mm256_min_ps(lhs.ymm, rhs.ymm)}; }
inline vec8f max(const vec8f& lhs, vec8f& rhs) { return {_mm256_max_ps(lhs.ymm, rhs.ymm)}; }

inline vec8f sqrt(const vec8f& x) { return {_mm256_sqrt_ps(x.ymm)}; }
// maximum relative error for this approximation is less than 1.5*2^-12
inline vec8f reciprocalSqrt(const vec8f& x) { return {_mm256_sqrt_ps(x.ymm)}; }

inline vec8f addSub(const vec8f& lhs, const vec8f& rhs) { return {_mm256_addsub_ps(lhs.ymm, rhs.ymm)}; }

// maximum relative error for this approximation is less than 1.5*2^-12
inline vec8f reciprocal(const vec8f& x) { return {_mm256_rcp_ps(x.ymm)}; }

// horizontal operations; use if you have to
inline vec8f hAdd(const vec8f& lhs, const vec8f& rhs) { return {_mm256_hadd_ps(lhs.ymm, rhs.ymm)}; }
inline vec8f hSub(const vec8f& lhs, const vec8f& rhs) { return {_mm256_hsub_ps(lhs.ymm, rhs.ymm)}; }
inline vec8f hEvenDup(const vec8f& x) { return {_mm256_moveldup_ps(x.ymm)}; }
inline vec8f hOddDup(const vec8f& x) { return {_mm256_movehdup_ps(x.ymm)}; }

template <int MultiplyMask8Bit = 0xF0, int StoreMask8Bit = 0x0F>
inline vec8f dot(const vec8f& lhs, const vec8f& rhs) { return {_mm256_dp_ps(lhs.ymm, rhs.ymm, MultiplyMask8Bit | StoreMask8Bit)}; }

// masks
inline int moveMask(const vec8f& x) { return _mm256_movemask_ps(x.ymm); }

// tests
inline bool isZFlagSet(const vec8f& lhs, const vec8f& rhs) { return _mm256_testz_ps(lhs.ymm, rhs.ymm) != 0; }
inline bool isCFlagSet(const vec8f& lhs, const vec8f& rhs) { return _mm256_testc_ps(lhs.ymm, rhs.ymm) != 0; }
inline bool isZAndCFlagClear(const vec8f& lhs, const vec8f& rhs) { return _mm256_testnzc_ps(lhs.ymm, rhs.ymm) != 0; }


// FMA
// fusedMulAdd: (a * b) + c
inline vec8f fusedMulAdd(const vec8f& a, const vec8f& b, const vec8f& c) { return {_mm256_fmadd_ps(a.ymm, b.ymm, c.ymm)}; }

// fusedMulSub: (a * b) - c
inline vec8f fusedMulSub(const vec8f& a, const vec8f& b, const vec8f& c) { return {_mm256_fmsub_ps(a.ymm, b.ymm, c.ymm)}; }

// fusedMulNegateAdd: -(a * b) + c
inline vec8f fusedMulNegateAdd(const vec8f& a, const vec8f& b, const vec8f& c) { return {_mm256_fnmadd_ps(a.ymm, b.ymm, c.ymm)}; }

// fusedMulNegateSub: -(a * b) - c
inline vec8f fusedMulNegateSub(const vec8f& a, const vec8f& b, const vec8f& c) { return {_mm256_fnmsub_ps(a.ymm, b.ymm, c.ymm)}; }

// fusedMulAddSub: (a * b) -+ c;  even: -, odd: +
inline vec8f fusedMulAddSub(const vec8f& a, const vec8f& b, const vec8f& c) { return {_mm256_fmaddsub_ps(a.ymm, b.ymm, c.ymm)}; }

// fusedSubAdd: (a * b) +- c;  even: +, odd: -
inline vec8f fusedMulSubAdd(const vec8f& a, const vec8f& b, const vec8f& c) { return {_mm256_fmsubadd_ps(a.ymm, b.ymm, c.ymm)}; }

}
