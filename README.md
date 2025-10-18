# CUDA Kernels

Some experiments on writing CUDA kernels.

Currently only tested on Windows.

## Building

From this directory:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

This will make an `examples.exe` executable in build/Release

## Testing

Using GoogleTest for testing kernels.

From this directory:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

cd build
ctest --output-on-failure
```
