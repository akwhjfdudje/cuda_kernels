# CUDA Kernels

Some experiments on writing CUDA kernels.

## Building

In root directory:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The compiled kernels will be in `cuda_kernels.lib`

This will also generate an `examples.exe` executable in build/Release (check src/examples.cpp):

```
.\build\Release\examples.exe
```

## Testing

Using GoogleTest for testing kernels.

In root directory:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

cd build
ctest --output-on-failure
```

## Documentation

In root directory:
```
mkdir docs
mkdir docs/doxygen
doxygen
```

To regenerate:
```
doxygen
```
