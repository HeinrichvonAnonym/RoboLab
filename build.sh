# check if build directory exists
if [ ! -d "build" ]; then
    mkdir build
fi

# build the project
cd build
cmake ..
make
cd ..