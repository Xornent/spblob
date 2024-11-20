
cpp = g++
cc  = gcc

# this preserves debug symbols into the executable file.
# needed for debugging using gdb.

# debug = -g -O0 -fno-inline

# if not for debug, we should use -O2 flag, this can greatly improve the speed
# of blobshed routine.

debug = -O2

lib = $(shell pkg-config --libs opencv4)
inc = $(shell pkg-config --cflags opencv4)

all: blobroi blobshed
all-win: blobroi-win blobshed-win

blobroi: blobroi.cpp blobroi.h blob.cpp blob.h
	$(cpp) blob.cpp blobroi.cpp blobroi.h blob.h $(inc) $(lib) -o blobroi -Dunix $(debug)

blobroi-win: blobroi.cpp blobroi.h blob.cpp blob.h
	$(cpp) blob.cpp blobroi.cpp blobroi.h blob.h $(inc) $(lib) -o blobroi $(debug)

blobshed: blobshed.cpp blobshed.h blob.cpp blob.h
	$(cpp) blob.cpp blobshed.cpp blobshed.h blob.h $(inc) $(lib) -o blobshed -Dunix $(debug)

blobshed-win: blobshed.cpp blobshed.h blob.cpp blob.h
	$(cpp) blob.cpp blobshed.cpp blobshed.h blob.h $(inc) $(lib) -o blobshed $(debug)
