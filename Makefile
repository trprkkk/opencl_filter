# Convenience build/test entry points (no OpenCL needed for the reference test).
CC ?= g++
CXXFLAGS := -O2 -std=c++17 -Wall

REF_BIN := build/ktgmc_cpu_ref

.PHONY: all test lint ref clean

all: test lint

lint:
	./lint/lint_opencl.sh

ref:
	mkdir -p build
	$(CXX) $(CXXFLAGS) -w sim/ktgmc_cpu_ref.cpp -o $(REF_BIN)

test: ref lint
	python3 python/run_validation.py
	python3 python/run_motion_core.py
	python3 python/run_mv_aux.py
	python3 python/run_mv_interp.py
	python3 python/run_mv_mean.py
	python3 python/run_mv_searchprep.py
	python3 python/run_mv_io.py

clean:
	rm -rf build
