
cpp = g++
cc  = gcc

# debug = -g -O0 -fno-inline
debug = -O3

lib = -lopencv_gapi -lopencv_stitching -lopencv_alphamat -lopencv_aruco \
      -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv \
	  -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_face \
	  -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs \
	  -lopencv_img_hash -lopencv_intensity_transform -lopencv_line_descriptor \
	  -lopencv_mcc -lopencv_quality -lopencv_rapid -lopencv_reg -lopencv_rgbd \
	  -lopencv_saliency -lopencv_shape -lopencv_stereo \
	  -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres \
	  -lopencv_optflow -lopencv_surface_matching -lopencv_tracking \
	  -lopencv_highgui -lopencv_datasets -lopencv_text -lopencv_plot \
	  -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz \
	  -lopencv_wechat_qrcode -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect \
	  -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d \
	  -lopencv_dnn -lopencv_flann -lopencv_xphoto -lopencv_photo \
	  -lopencv_imgproc -lopencv_core

all: spblob # spblob.ch spblob.orig

# india, default
spblob: blob.cpp blob.h libdistrib.a
	$(cpp) blob.cpp blob.h libdistrib.a -I/usr/include/opencv4 $(lib) -o spblob $(debug)

# spblob.ch is a parameter adjustment for ch.papers.
# with grayish beginnings and extra lengths
# spblob.ch: blob.ch.cpp blob.h libdistrib.a
# 	$(cpp) blob.ch.cpp blob.h libdistrib.a -I/usr/include/opencv4 $(lib) -o spblob.ch $(debug)

# to deal with non-india/china's primitive samples, where tags are placed nearby
# spblob.orig: blob.orig.cpp blob.h libdistrib.a
# 	$(cpp) blob.orig.cpp blob.h libdistrib.a -I/usr/include/opencv4 $(lib) -o spblob.orig $(debug)

bratio.o: bratio.c distrib.h 
	$(cc) bratio.c -lm -c -o bratio.o -fcompare-debug-second -w $(debug)

distrib.o: distrib.c bratio.c distrib.h
	$(cc) distrib.c -lm -c -o distrib.o -fcompare-debug-second -w $(debug)

libdistrib.a: bratio.o distrib.o
	ar -rc libdistrib.a bratio.o distrib.o