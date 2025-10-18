# CUDA Kernels

Some experiments on writing CUDA kernels 

Future work involves putting these into a library for use with other projects

## Building

Currently only tested on Windows

From this directory:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

This will make an example executable in build/Release

## Testing

Using GoogleTest for testing kernels.

From this directory:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

cd build
ctest --output-on-failure
```
