
cpp = g++
cc  = gcc

# debug = -g -O0 -fno-inline
debug = -O2

lib = $(shell pkg-config --libs opencv)
inc = $(shell pkg-config --cflags opencv)

all: spblob

spblob: blob.cpp blob.h libdistrib.a
	$(cpp) blob.cpp blob.h libdistrib.a $(inc) $(lib) -o spblob $(debug)

bratio.o: bratio.c distrib.h 
	$(cc) bratio.c -lm -c -o bratio.o -fcompare-debug-second -w $(debug)

distrib.o: distrib.c bratio.c distrib.h
	$(cc) distrib.c -lm -c -o distrib.o -fcompare-debug-second -w $(debug)

libdistrib.a: bratio.o distrib.o
	ar -rc libdistrib.a bratio.o distrib.o