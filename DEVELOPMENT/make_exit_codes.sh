#!/bin/bash


# e.g.
awk 'BEGIN{n=10} /sys\.exit/{gsub("[1-9][0-9]*",++n)} {print}' SHARC_MOLCAS.py > SHARC_test.py