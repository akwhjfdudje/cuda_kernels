# CUDA Kernels

Some experiments on writing CUDA kernels 

Future work involves putting these into a library for use with other projects

## Building

Currently only tested on Windows

From this directory:
```
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

This will make an example executable in build/Release
