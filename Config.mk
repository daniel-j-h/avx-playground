CC = gcc
CXX = g++
CXXFLAGS = -pthread -fdiagnostics-color=auto -fmax-errors=1 -std=c++14 -O2 -g -Wall -Wextra -pedantic -Wuninitialized -Wstrict-overflow=3 -Wshadow
CXXFLAGS += -mavx -mavx2 -mfma
LDLIBS = -lstdc++ -lpthread
LDFLAGS += -Wl,-O1 -Wl,--hash-style=gnu -Wl,--sort-common -Wl,--demangle -Wl,--build-id
