# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 11:31:57 2018

@author: Main
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("base", type=float, help="the base")
parser.add_argument("exponent", type=float, help="the exponential factor")
parser.add_argument("-v", "--verbose", action="count", default=0, help="change the verbosity of output")

args = parser.parse_args()

answer = args.base ** args.exponent

if args.verbose >= 2:
    print("Running '{}'".format(__file__))
if args.verbose >= 1:
    print("{} ^ {} = ".format(args.base, args.exponent, answer), end='')
print(answer)