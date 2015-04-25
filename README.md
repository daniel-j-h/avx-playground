# AVX/AVX2/FMA Playground

Playground for learning AVX/AVX2/FMA on my Broadwell Laptop.
Abstractions for Vec8Float, Vec8Int in C++14.

There's so much different in floating point values vs. integer values in AVX, that there is no common API.
Some functionality is there for both types such as initialization, loading, storing, and so on.
But there is e.g. no division for integer types and no streamed load for floating point types in AVX, just to name two exceptions.

Disclaimer: do not use this for anything serious.


## Detect.s

x86-64 assembly to detect AVX/AVX2/FMA feature (cpuid) and OS support for saving and restoring register state (xgetbv).


## Vec8Float

8 x 32bit single precision floating point values


## Vec8Int

8 x 32bit signed integer values


## License

Copyright © 2015 Daniel J. Hofmann

Distributed under the MIT License (MIT).
