#!/usr/bin/env python3
import time
from dolphin import DolphinRunner
from argparse import ArgumentParser
from multiprocessing import Process
from cpu import CPU
import util
import tempfile

def main(args):
    params = {}
    util.update(params, **args.__dict__)
    print(params)

    if params['gui']:
      params['dolphin'] = True

    if params['user'] is None:
      params['user'] = tempfile.mkdtemp() + '/'

    print("Creating cpu.")
    cpu = CPU(**params)

    params['cpus'] = cpu.pids

    if params['dolphin']:
      dolphinRunner = DolphinRunner(**params)
      # delay for a bit to let the cpu start up
      time.sleep(5)
      print("Running dolphin.")
      dolphin = dolphinRunner()
    else:
      dolphin = None

    return cpu, dolphin

