#!/usr/bin/env python3
import os

global DEBUG
DEBUG = False
if 'DEBUG' in os.environ and os.environ["DEBUG"].lower() in ["true", "false"]:
    DEBUG = os.environ["DEBUG"] == "true"

global PRINT
PRINT = True
if 'PRINT' in os.environ and os.environ["PRINT"].lower() in ["true", "false"]:
    PRINT = os.environ["PRINT"] == "true"
