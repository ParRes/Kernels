#!/usr/bin/env python
import sys
import os
import subprocess

def runCommand(command, logFileName):
    print command
#    subprocess.call(command + " >> " + logFileName, shell=True)

from optparse import OptionParser          # requires python 2.3
from socket import gethostname

#
# Now the job specific code.
# 
def generateLogName(testName):
    import datetime

    hostname = os.uname()[1]
    today    = datetime.date.today().strftime("%d%b%y")
    fileRoot = hostname + "-"+today+"-"+testName
    filename = fileRoot+".out"

    # Add a file version number if necessary
    delta = 1
    while os.path.exists(filename):
        filename = fileRoot+"-"+str(delta)+".out"
        delta += 1

    return filename

# KMP_HW_SUBSET=1T seems to work to get one thread/core across multiple sockets on Xeon
# and on KNL, to use all available cores, and to step between the sockets in the expected place
# (i.e. when there's one thread on each core in a socket).
affinity = "KMP_AFFINITY=compact,granularity=fine "

hints    = ("none", "uncontended", "contended", "speculative")
threads  = (1,2)
maxCores = 4*28

for hint in hints:
    for threadsPerCore in threads:
        logFile = generateLogName("refcount-"+str(threadsPerCore)+"T-"+hint)
        print "Writing "+logFile
        env = "KMP_HW_SUBSET="+str(threadsPerCore)+"T " + affinity
        for coreCount in [1,2,4,8] + range(16,maxCores+8,8) + ([] if maxCores%8==0 else [maxCores,]):
            if coreCount>maxCores:
                continue        # Could be smarter, but really this doesn't matter at all
            command = env+"./refcount " + str(coreCount*threadsPerCore) + " 10000000 1000 "+hint
            runCommand (command, logFile)
            
