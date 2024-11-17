
cpp = g++
cc  = gcc

# debug = -g -O0 -fno-inline
debug = -O2

lib = $(shell pkg-config --libs opencv4)
inc = $(shell pkg-config --cflags opencv4)

all: spblob

spblob: blob.cpp blob.h
	$(cpp) blob.cpp blob.h $(inc) $(lib) -o spblob -Dunix $(debug)

spblob-win: blob.cpp blob.h libdistrib.a
	$(cpp) blob.cpp blob.h $(inc) $(lib) -o spblob $(debug)
