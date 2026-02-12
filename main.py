import sys
#import runpy

import nz.run_processing_db as rpdb

def run():
    print("Starting ...")

    run_name="__main__"
    rpdb.run()
    # Auto exec
    #runpy.run_module("nz.run_processing_db", run_name=run_name)



if __name__ == "__main__":

    print("===================================")

    run()

    print("===================================")



