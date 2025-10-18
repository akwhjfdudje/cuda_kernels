# CUDA Kernels

Some experiments on writing CUDA kernels.

Currently only tested on Windows.

## Building

From this directory:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The compiled kernels will be in `cuda_kernels.lib`

This will also generate an `examples.exe` executable in build/Release:

```
.\build\Release\examples.exe
```

## Testing

Using GoogleTest for testing kernels.

From this directory:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

cd build
ctest --output-on-failure
```
