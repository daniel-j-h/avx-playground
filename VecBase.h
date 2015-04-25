#pragma once

#include <cstddef>
#include <iterator>
#include <algorithm>

namespace avx {

// specialized for valid combinations
template <typename T, std::size_t N>
struct vec final {};


// compiler-generated operator<< for _all_ vec<T, N> combinations
template <typename OutStream, typename T, std::size_t N>
OutStream& operator<<(OutStream& o, const vec<T, N>& v) {
  std::copy_n(begin(v), N, std::ostream_iterator<T>(o, " "));
  return o;
}


// compiler-generated begin,end for _all_ vec<T, N> combinations
template <typename T, std::size_t N>
inline T* begin(vec<T, N>& x) {
  return &x[0];
}

template <typename T, std::size_t N>
inline const T* begin(const vec<T, N>& x) {
  return &x[0];
}

template <typename T, std::size_t N>
inline T* end(vec<T, N>& x) {
  return begin(x) + N;
}

template <typename T, std::size_t N>
inline const T* end(const vec<T, N>& x) {
  return begin(x) + N;
}


// dependency breakers
void zeroUpper() { _mm256_zeroupper(); }
void zeroAll() { _mm256_zeroall(); }


// TODO(daniel): strided iterators: vec_iterator<T,N>, make_vec_iterator<T,N>() -- iterator adaptors awful to implement
// :(
}
