#!/bin/bash
cd /user/lorenz/bin/sharc/develop/source/ &&
make clean
cd ../pysharc/ &&
echo "asdf"
make clean 
make install &&
cd ../source/ &&
make install
