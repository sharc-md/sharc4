#!/bin/bash

awk 'BEGIN{n=10} /sys\.exit/{gsub("[1-9][0-9]*",++n)} {print}' $1 > $1.subst
grep -B 1 'sys.exit' $1 