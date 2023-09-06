#!/usr/bin/env python3

import sys
import logging
import os
SHARCPRINT = 11


class CustomFormatter(logging.Formatter):
    err_fmt = "ERROR: %(message)s"
    dbg_fmt = "DEBUG: [%(filename)s:%(funcName)s():%(lineno)s]%(message)s"
    info_fmt = "%(message)s"
    warn_fmt = "WARNING: %(message)s"

    def format(self, record):
        # Replace the original format with one customized by logging level
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

        # Call the original formatter class to do the grunt work
        formatter = logging.Formatter(self._fmt)

        return formatter.format(record)





fmt = CustomFormatter()
hdlr = logging.StreamHandler(sys.stdout)
hdlr._name = 'rootHandler'

hdlr.setFormatter(fmt)
logging.root.handlers = []
logging.root.addHandler(hdlr)
logging.addLevelName(SHARCPRINT, "SHARCPRINT")
logging.SHARCPRINT = SHARCPRINT

envlevel = os.environ.get('SHARCLOG')
match envlevel:
    case 'DEBUG':
        loglevel = logging.DEBUG
    case 'INFO':
        loglevel = logging.INFO
    case 'PRINT':
        loglevel = logging.SHARCPRINT
    case 'ERROR':
        loglevel = logging.ERROR
    case 'WARNING':
        loglevel = logging.WARNING
    case None:
        loglevel = logging.INFO
    case _:
        loglevel = logging.INFO

logging.root.setLevel(loglevel)


def sharcprint(msg, *args, **kwargs):
    """
    Log 'msg % args' with severity 'SHARCPRINT'.

    To pass exception information, use the keyword argument exc_info with
    a true value, e.g.

    logger.print("Houston, we have a %s", "interesting problem", exc_info=1)
    """
    logging.log(SHARCPRINT, msg, *args, **kwargs)



logging.print = sharcprint
log = logging
if __name__ == "__main__":
    print("Test logger")
    log.root.setLevel(log.SHARCPRINT)
    log.error("Error")
    log.warning("warning")
    log.info("info")
    log.print("print")
    log.debug("debug")
