#!/usr/bin/env bash

rm build/ -rf
mkdir -p build
pushd build
cmake ..
make -j 4
popd
