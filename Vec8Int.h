#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <immintrin.h>
#include "VecBase.h"

namespace avx {

using vec8i = vec<std::int32_t, 8u>;

template <>
struct vec<std::int32_t, 8u> final {
  using Value = std::int32_t;
  static const constexpr std::size_t Size = 8u;

  // thin abstraction; instead of explicit conversion operator and the need for static-casting, just use .ymm
  __m256i ymm;

  vec() : ymm(_mm256_undefined_si256()) {}
  vec(__m256i x) : ymm(x) {}
  vec(Value scalar) { load(scalar); }
  vec(const Value* first, const Value* last) { load(first, last); }
  vec(Value e7, Value e6, Value e5, Value e4, Value e3, Value e2, Value e1, Value e0) {
    load(e7, e6, e5, e4, e3, e2, e1, e0);
  }

  // load
  void load(Value scalar) { ymm = _mm256_set1_epi32(scalar); }

  void load(const Value* first, const Value* last) {
    assert(last - first == Size);
    ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(first));
    // ymm = _m256_lddqu_si256(first);  may be faster in cases when the integers are not cache line boundary aligned
  }

  void load_aligned(const Value* first, const Value* last) {
    assert(last - first == Size);
    ymm = _mm256_load_si256(reinterpret_cast<const __m256i*>(first));
  }

  // AVX2, non-temporal memory hint
  void load_aligned_stream(const Value* first, const Value* last) {
    assert(last - first == Size);
    ymm = _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(first));
  }

  void load(Value e7, Value e6, Value e5, Value e4, Value e3, Value e2, Value e1, Value e0) {
    ymm = _mm256_setr_epi32(e7, e6, e5, e4, e3, e2, e1, e0);
  }

  // store
  void store(Value* first, Value* last) {
    assert(last - first == Size);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(first), ymm);
  }

  void store_aligned(Value* first, Value* last) {
    assert(last - first == Size);
    _mm256_store_si256(reinterpret_cast<__m256i*>(first), ymm);
  }

  void stream_aligned(Value* first, Value* last) {
    assert(last - first == Size);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(first), ymm);
  }

  // misc
  void zero() { ymm = _mm256_setzero_si256(); }

  // compound ops, no implicit conversion needed for lhs
  vec8i& operator+=(const vec8i& rhs) {
    ymm = _mm256_add_epi32(ymm, rhs.ymm);
    return *this;
  }

  vec8i& operator-=(const vec8i& rhs) {
    ymm = _mm256_sub_epi32(ymm, rhs.ymm);
    return *this;
  }

  vec8i& operator*=(const vec8i& rhs) {
    ymm = _mm256_mul_epi32(ymm, rhs.ymm);
    return *this;
  }

  // there is no div mmnemonic for integers

  vec8i& operator|=(const vec8i& rhs) {
    ymm = _mm256_or_si256(ymm, rhs.ymm);
    return *this;
  }

  vec8i& operator&=(const vec8i& rhs) {
    ymm = _mm256_and_si256(ymm, rhs.ymm);
    return *this;
  }

  vec8i& operator^=(const vec8i& rhs) {
    ymm = _mm256_xor_si256(ymm, rhs.ymm);
    return *this;
  }

  vec8i& operator<<=(const vec8i& rhs) {
    ymm = _mm256_sllv_epi32(ymm, rhs.ymm);
    return *this;
  }

  vec8i& operator>>=(const vec8i& rhs) {
    ymm = _mm256_srlv_epi32(ymm, rhs.ymm);
    return *this;
  }

  // subscript
  Value& operator[](std::size_t index) {
    assert(index < Size);
    return *(reinterpret_cast<Value*>(&ymm) + index);
  }

  const Value& operator[](std::size_t index) const {
    assert(index < Size);
    return *(reinterpret_cast<const Value*>(&ymm) + index);
  }
};


// free standing functions, participate in implicit conversion for lhs and rhs
inline vec8i operator+(const vec8i& lhs, const vec8i& rhs) { return {_mm256_add_epi32(lhs.ymm, rhs.ymm)}; }
inline vec8i operator-(const vec8i& lhs, const vec8i& rhs) { return {_mm256_sub_epi32(lhs.ymm, rhs.ymm)}; }
inline vec8i operator*(const vec8i& lhs, const vec8i& rhs) { return {_mm256_mul_epi32(lhs.ymm, rhs.ymm)}; }
// there is no div mnemonic for integers

// TODO(daniel): andnot, op%, op!, ++, --
inline vec8i operator|(const vec8i& lhs, const vec8i& rhs) { return {_mm256_or_si256(lhs.ymm, rhs.ymm)}; }
inline vec8i operator&(const vec8i& lhs, const vec8i& rhs) { return {_mm256_and_si256(lhs.ymm, rhs.ymm)}; }
inline vec8i operator^(const vec8i& lhs, const vec8i& rhs) { return {_mm256_xor_si256(lhs.ymm, rhs.ymm)}; }

// unary
inline vec8i operator~(const vec8i& x) { return {_mm256_xor_si256(x.ymm, _mm256_set1_epi32(-1))}; }
inline vec8i operator-(const vec8i& x) { return {_mm256_sub_epi32(_mm256_set1_epi32(0), x.ymm)}; }

// zero extend; see explicit shifting functions below
inline vec8i operator<<(const vec8i& lhs, const vec8i& rhs) { return {_mm256_sllv_epi32(lhs.ymm, rhs.ymm)}; }
inline vec8i operator>>(const vec8i& lhs, const vec8i& rhs) { return {_mm256_srlv_epi32(lhs.ymm, rhs.ymm)}; }

inline vec8i operator==(const vec8i& lhs, const vec8i& rhs) { return {_mm256_cmpeq_epi32(lhs.ymm, rhs.ymm)}; }
inline vec8i operator>(const vec8i& lhs, const vec8i& rhs) { return {_mm256_cmpgt_epi32(lhs.ymm, rhs.ymm)}; }

inline vec8i operator!=(const vec8i& lhs, const vec8i& rhs) { return ~(lhs == rhs); }
inline vec8i operator<(const vec8i& lhs, const vec8i& rhs) { return rhs > lhs; }
inline vec8i operator>=(const vec8i& lhs, const vec8i& rhs) { return ~(lhs < rhs); }
inline vec8i operator<=(const vec8i& lhs, const vec8i& rhs) { return ~(lhs > rhs); }


// explicit shifting -- there is no slai mnemonic, shiftLeftSignExtend that is
template <int ShiftMask8Bit>
inline vec8i shiftRightZeroExtend(const vec8i& x) { return {_mm256_srli_epi32(x.ymm, ShiftMask8Bit)}; }

template <int ShiftMask8Bit>
inline vec8i shiftLeftZeroExtend(const vec8i& x) { return {_mm256_slli_epi32(x.ymm, ShiftMask8Bit)}; }

template <int ShiftMask8Bit>
inline vec8i shiftRightSignExtend(const vec8i& x) { return {_mm256_srai_epi32(x.ymm, ShiftMask8Bit)}; }


// misc
template <int BlendMask8Bit>
inline vec8i blend(const vec8i& lhs, const vec8i& rhs) { return {_mm256_blend_epi32(lhs.ymm, rhs.ymm, BlendMask8Bit)}; }

inline vec8i permute(const vec8i& x, const vec8i& mask) { return {_mm256_permutevar8x32_epi32(x.ymm, mask.ymm)}; }

template <int ShuffleMask8Bit>
inline vec8i shuffle(const vec8i& x) { return {_mm256_shuffle_epi32(x.ymm, ShuffleMask8Bit)}; }

inline vec8i unpackHigh(const vec8i& lhs, const vec8i& rhs) { return {_mm256_unpackhi_epi32(lhs.ymm, rhs.ymm)}; }
inline vec8i unpackLow(const vec8i& lhs, const vec8i& rhs) { return {_mm256_unpacklo_epi32(lhs.ymm, rhs.ymm)}; }


// math
inline vec8i min(const vec8i& lhs, vec8i& rhs) { return {_mm256_min_epi32(lhs.ymm, rhs.ymm)}; }
inline vec8i max(const vec8i& lhs, vec8i& rhs) { return {_mm256_max_epi32(lhs.ymm, rhs.ymm)}; }

inline vec8i abs(const vec8i& x) { return {_mm256_abs_epi32(x.ymm)}; }

// horizontal operations; use if you have to
inline vec8i hAdd(const vec8i& lhs, const vec8i& rhs) { return {_mm256_hadd_epi32(lhs.ymm, rhs.ymm)}; }
inline vec8i hSub(const vec8i& lhs, const vec8i& rhs) { return {_mm256_hsub_epi32(lhs.ymm, rhs.ymm)}; }

// tests
inline bool isZFlagSet(const vec8i& lhs, const vec8i& rhs) { return _mm256_testz_si256(lhs.ymm, rhs.ymm) != 0; }
inline bool isCFlagSet(const vec8i& lhs, const vec8i& rhs) { return _mm256_testc_si256(lhs.ymm, rhs.ymm) != 0; }
inline bool isZAndCFlagClear(const vec8i& lhs, const vec8i& rhs) { return _mm256_testnzc_si256(lhs.ymm, rhs.ymm) != 0; }

// TODO(daniel): gather, masks
}
