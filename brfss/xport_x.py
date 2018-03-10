import xport
from xport import Reader as RR 

XPT_FILE = '/Users/amitn/Documents/ml_codes/brfss/LLCP2016.XPT'

x = 0

with open(XPT_FILE, 'rb') as f:
    for row in xport.Reader(f):
        print(row)
        x += 1
        if x == 3:
        	break