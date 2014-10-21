#!/bin/bash

# for i in *.pdf;
# do
# 
#   name=$(echo $i | sed 's/.pdf//')
#   convert -density 600 $name.pdf $name.png
# 
# done

for i in *.png;
do

  convert -resize 25% $i temp.png
  mv temp.png $i

done