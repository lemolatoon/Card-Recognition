MKFILE_PATH = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: setupcv
setupcv:
	sudo apt update
	sudo apt intall -y cmake clang libclang-dev g++ git pkg-config libopencv-dev

buildcv:
	mkdir -p build
	cd build && \
	git clone https://github.com/opencv/opencv.git && \
	cmake ../opencv \
	cmake --build .
