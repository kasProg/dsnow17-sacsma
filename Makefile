.PHONY: build build-checked test clean

build:
	./fortran/build.sh

build-checked:
	SNOW17_BOUNDS_CHECK=1 ./fortran/build.sh

test: build
	python3 -m pytest tests/ -v

clean:
	rm -f fortran/*.so fortran/*.o *.mod
