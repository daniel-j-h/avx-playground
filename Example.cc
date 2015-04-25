#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <chrono>

#include "Vec.h"

#define R(...) std::cout << __VA_ARGS__ << std::endl;


void vecPerf() {
  using clock = std::chrono::high_resolution_clock;
  avx::vec8f z(0.f);

  std::vector<float> fst(1'000'000'000u);
  std::vector<float> snd(1'000'000'000u);

  const auto t0 = clock::now();
  std::iota(begin(fst), end(fst), 0u);
  std::iota(snd.rbegin(), snd.rend(), 0u);
  const auto t1 = clock::now();
  R(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

  R("go");
  const auto t2 = clock::now();

  for (std::size_t i{0}; i < fst.size(); i += 8) {
    avx::vec8f a(&fst[i], &fst[i + 8]);
    avx::vec8f b(&snd[i], &snd[i + 8]);
    z += a + ceil(b);
    z.store(&fst[i], &fst[i + 8]);
  }

  R(fst.front());
  R(fst.back());

  const auto t3 = clock::now();
  R(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count());

  R(z);
}


void blendTest() {
  avx::vec8f a{0.};
  avx::vec8f b{1.};
  R(a);
  R(b);

  const constexpr auto oddEvenMask = 1 << 1 | 1 << 3 | 1 << 5 | 1 << 7;
  const constexpr auto evenOddMask = 1 << 0 | 1 << 2 | 1 << 4 | 1 << 6;
  const constexpr auto selectFirstMask = 0b1000'0000;
  const constexpr auto selectLastMask = 0b0000'0001;
  R(avx::blend<oddEvenMask>(a, b));
  R(avx::blend<evenOddMask>(a, b));
  R(avx::blend<selectFirstMask>(a, b));
  R(avx::blend<selectLastMask>(a, b));
}


void dotTest() {
  avx::vec8f a{1, 1, 1, 1, 2, 2, 2, 2};
  avx::vec8f b{2, 2, 2, 2, 3, 3, 3, 3};
  R(a);
  R(b);

  // dot really is dot(a[0:4],b[0:4]) ++ dot(a[4:8],b[4:8]), resulting in effectively two dot products instead of one
  const auto dp = dot(a, b);
  R(dp);
}


void permuteTest() {
  avx::vec8f v{1, 2, 3, 4, 5, 6, 7, 8};
  R(v);
  // select4 on each 128 lane
  R(avx::permute<0b1010'1010>(v));
  R(avx::permute<0b0101'0101>(v));

  avx::vec8f w{8, 7, 6, 5, 4, 3, 2, 1};
  R(v);
  R(w);
  R(avx::permute<0b0111'0111>(v, w));
}


void shuffleTest() {
  avx::vec8f x{10, 20, 30, 40, 50, 60, 70, 80};
  avx::vec8f y{11, 21, 31, 41, 51, 61, 71, 81};
  R(x);
  R(y);

  R(avx::shuffle<0b0101'0101>(x, y));
}


void unpackTest() {
  avx::vec8f x{10, 20, 30, 40, 50, 60, 70, 80};
  avx::vec8f y{11, 21, 31, 41, 51, 61, 71, 81};

  R(x);
  R(y);
  R(unpackHigh(x, y));
  R(unpackLow(x, y));
}


void fmaTest() {
  avx::vec8f a{10, 20, 30, 40, 50, 60, 70, 80};
  avx::vec8f b{0, 1, 0, 0, 0, 0, 0, 0};
  avx::vec8f c{1, 2, 3, 4, 5, 6, 7, 8};
  R(a);
  R(b);
  R(c);
  // a*b + c
  R(fusedMulAdd(a, b, c));
}


void initTest() {
  std::vector<std::int32_t> fst(1'000'000'000);
  std::vector<std::int32_t> snd(1'000'000'000);
  std::iota(begin(fst), end(fst), 0u);
  std::iota(snd.rbegin(), snd.rend(), 0u);

  avx::vec8i z{0};

  for (std::size_t i{0}; i < fst.size(); i += 8) {
    avx::vec8i a(&fst[i], &fst[i + 8]);
    avx::vec8i b(&snd[i], &snd[i + 8]);
    z += a * b;
    z.store(&fst[i], &fst[i + 8]);
  }

  R(fst.front() << ' ' << fst.back());
}


void shiftTest() {
  avx::vec8i v{1,2,3,4,5,6,7,8};
  R(v);

  auto l = v << 2;
  auto r = v >> 2;

  R(l);
  R(r);
}


void bitTest() {
  avx::vec8i x{0xFFFFFF};
  avx::vec8i y{0x0};

  R(x);
  R(y);
  R(~x);
  R(~y);
  R(0-x);
  R(abs(x));
}


void comparisonTest() {
  avx::vec8i a{1,2,3,4,5,6,7,8};
  avx::vec8i b{0,2,3,4,5,6,7,8};

  R((a > a));
  R((a < a));
  R((a >= a));
  R((a <= a));
  R((a != a));
  R((a == a));

  R((a < b));
  R((a > b));
}


int main() try {
  // vecPerf();
  // blendTest();
  // dotTest();
  // permuteTest();
  // shuffleTest();
  // unpackTest();
  // fmaTest();

  // initTest();
  // shiftTest();
  // bitTest();
  // comparisonTest();

} catch (const std::exception& e) {
  std::cerr << e.what() << std::endl;
}
