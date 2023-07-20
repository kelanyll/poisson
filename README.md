# Predicting goals scored with Poisson regression written in C++
This is a toy project predict goals scored using Poisson Regression on the Premier League 11/12 season. It uses C++, unconventionally, instead of the industry standard for similar use cases - Python and R.

## Coding standards and principles
As this was in part written to improve my C++ skills, I tried to follow a set of principles:
- Ensure we're using move semantics wherever possible. This can be useful to reduce the instances where we're copying heavy 2d data structures.
- Use test-driven development. Although the main reason for this is that it shortens the feedback loop.
- Follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es23-prefer-the--initializer-syntax) and prefer uniform initialization.

## Development
This was developed on an x64-linux machine with dependencies on dataframe@1.22.0 and gtest@1.13.0.

We build the BOOM library from source and it's currently locked to [this](https://github.com/steve-the-bayesian/BOOM/commit/2ff87325fbd2e3178b4c54d6508131a2fbfef90e) commit.

If using vcpkg to manage dependencies, run `cmake` with `-DCMAKE_TOOLCHAIN_FILE=/usr/local/vcpkg/scripts/buildsystems/vcpkg.cmake`. If `CMakeCache.txt` doesn't have any vcpkg variables, delete it and generate the build again.