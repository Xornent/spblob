
cpp = g++
cc  = gcc

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

all: spblob

spblob: blob.cpp blob.h libdistrib.a
	$(cpp) blob.cpp blob.h libdistrib.a -I/usr/include/opencv4 $(lib) -o spblob

bratio.o: bratio.c distrib.h 
	$(cc) bratio.c -lm -c -o bratio.o -fcompare-debug-second -w

distrib.o: distrib.c bratio.c distrib.h
	$(cc) distrib.c -lm -c -o distrib.o -fcompare-debug-second -w

libdistrib.a: bratio.o distrib.o
	ar -rc libdistrib.a bratio.o distrib.o