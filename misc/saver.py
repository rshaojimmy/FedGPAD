import numpy as np
import os
import ntpath
import time
from . import utils

class Saver():
    def __init__(self, args, logfilename):

        self.args = args

        self.save_file = os.path.join(args.results_path, 'log', logfilename)
        if not os.path.exists(self.save_file):
            utils.mkdirs(self.save_file) 

        self.log_name = os.path.join(self.save_file, 'loss_log.txt')

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    def print_current_errors(self, epoch, i, errors):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.5f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


        # save to the disk
    def print_config(self):
        opt = vars(self.args)
        file_name = os.path.join(self.save_file, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(opt.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

       

