
cc  = gcc

bratio.o: bratio.c distrib.h 
	$(cc) bratio.c -lm -c -o bratio.o -fcompare-debug-second -w $(debug)

distrib.o: distrib.c bratio.c distrib.h
	$(cc) distrib.c -lm -c -o distrib.o -fcompare-debug-second -w $(debug)

libdistrib.a: bratio.o distrib.o
	ar -rc libdistrib.a bratio.o distrib.o