#!/bin/bash
cd /user/lorenz/bin/sharc/develop/source/ &&
make clean
cd ../pysharc/ &&
make clean 
make install &&
cd ../source/ &&
make install
