#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************

import sys
import logging
import os
import re

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
SHARCPRINT = 11
DEBUG = 10
TRACE = 1
NOTSET = 0


class CustomFormatter(logging.Formatter):
    err_fmt = "ERROR: %(message)s"
    dbg_fmt = "DEBUG: [%(filename)s:%(funcName)s():%(lineno)s] %(message)s"
    info_fmt = "%(message)s"
    warn_fmt = "WARNING: %(message)s"
    trace_fmt = "TRACE: %(message)s"

    def format(self, record):
        # Replace the original format with one customized by logging level
        if hasattr(record, "simple") and record.simple:
            return record.getMessage()

        if record.levelno == logging.DEBUG:
            self._fmt = CustomFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._fmt = CustomFormatter.info_fmt

        elif record.levelno == SHARCPRINT:
            self._fmt = CustomFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = CustomFormatter.err_fmt

        elif record.levelno == logging.WARNING:
            self._fmt = CustomFormatter.warn_fmt

        elif record.levelno == logging.TRACE:
            self._fmt = CustomFormatter.trace_fmt

        # Call the original formatter class to do the grunt work
        formatter = logging.Formatter(self._fmt)

        return formatter.format(record)


fmt = CustomFormatter()
hdlr = logging.StreamHandler(sys.stdout)
hdlr._name = "rootHandler"

hdlr.setFormatter(fmt)
logging.root.handlers = []
logging.root.addHandler(hdlr)
logging.addLevelName(SHARCPRINT, "SHARCPRINT")
logging.SHARCPRINT = SHARCPRINT
logging.addLevelName(TRACE, "SHARCPRINT")
logging.TRACE = TRACE

envlevel = os.getenv("SHARCLOG")
if not envlevel:
    envlevel = os.getenv("SHARC_LOG")

loglevel = logging.INFO
match envlevel:
    case "TRACE":
        loglevel = logging.TRACE
    case "DEBUG":
        loglevel = logging.DEBUG
    case "INFO":
        loglevel = logging.INFO
    case "PRINT":
        loglevel = logging.SHARCPRINT
    case "ERROR":
        loglevel = logging.ERROR
    case "WARNING":
        loglevel = logging.WARNING
    case str() if int(envlevel) > 0:
        loglevel = int(envlevel)
    case _:
        loglevel = logging.INFO

# logging.root.setLevel(loglevel)


def sharcprint(msg, *args, **kwargs):
    """
    Log 'msg % args' with severity 'SHARCPRINT'.

    To pass exception information, use the keyword argument exc_info with
    a true value, e.g.

    logger.print("Houston, we have a %s", "interesting problem", exc_info=1)
    """
    logging.log(SHARCPRINT, msg, *args, **kwargs)


def trace(msg, *args, **kwargs):
    """
    Log 'msg % args' with severity 'TRACE'.

    use for exhaustive printing of runtime information

    To pass exception information, use the keyword argument exc_info with
    a true value, e.g.

    logger.trace("Houston, we have a %s", "interesting problem", exc_info=1)
    """
    logging.log(TRACE, msg, *args, **kwargs)


log = logging.getLogger("SHARC_LOGGER")
log.setLevel(loglevel)
log.print = sharcprint
log.trace = trace

if __name__ == "__main__":
    print("Test logger")
    log.root.setLevel(loglevel)
    log.error("Error")
    log.warning("warning")
    log.info("info")
    log.print("print")
    log.debug("debug")
    log.trace("trace")
