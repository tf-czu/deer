#!/usr/bin/python
"""
    usage:
         sudo python3 deer_detect.py label
"""

import sys
import datetime
import time
import os
import numpy as np
import pyaudio
from thread import start_new_thread
import picamera


CHUNK = 800 #1024
FORMAT = pyaudio.paInt16 #pyaudio.paInt16
CHANNELS = 1
RATE = 4000 #44100 96000
RADAR_LOG = datetime.datetime.now().strftime("logs/radar_%y%m%d_%H%M%S") + ".log"
IR_LOG = datetime.datetime.now().strftime("logs/ir_%y%m%d_%H%M%S") + ".log"
VIDEO_DIR = datetime.datetime.now().strftime("logs/video_%y%m%d_%H%M%S")
os.mkdir(VIDEO_DIR)
TIME_LIMIT = 10

g_radarEnd = False
g_irEnd = False


def waiting4finish():
    global g_radarEnd, g_irEnd
    while True:
        print( "waiting for finish radar, IR", g_radarEnd, g_irEnd)
        if g_radarEnd and g_irEnd:
            return True
        time.sleep(1)


def runRadar( startTime, endTime ):
    global g_radarEnd
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK )
    print( "* recording radar data")
    frames = []
    tId = []
    radarLog = open( RADAR_LOG, "w" )
    while time.time() < endTime:
        try:
            data = stream.read(CHUNK)
        except:
            data = "0"
            print("no data")
        tId.append(time.time() - startTime)
        frames.append(data)
        #continue
    
    print("* done recording")
    for ii, frame in enumerate( frames ):
        radarLog.write(str(tId[ii])+" ")
        if len(frame) > 1:
            radarLog.write( str( list( np.fromstring( frame, dtype=np.int16) ) ) )
        else:
            radarLog.write("[0]")
        radarLog.write( "\r\n" )
    radarLog.close()

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Time for saveing", time.time()-endTime)
    g_radarEnd = True
    
    return None


def runIR(startTime, endTime):
    global g_irEnd
    fifo = open('/var/run/mlx9062x.sock', 'r')
    time.sleep(0.1)
    irLog = open(IR_LOG, "w")
    irData = []
    tId = []
    while time.time() < endTime:
        ir_raw = fifo.read()
        if len(ir_raw) < 128:
            continue
        ir_last = ir_raw[-128:]
        tId.append(time.time() - startTime)
        # go all numpy on it
        ir = np.frombuffer(ir_last, np.uint16)
        irData.append(list(ir))
    
    fifo.close()
    for ii, irD in enumerate( irData ):
        irLog.write(str(tId[ii])+" ")
        irLog.write(str(irD))
        irLog.write( "\r\n" )
    irLog.close()
    
    g_irEnd = True


def deerDetect( ):
    startTime = time.time()
    endTime = startTime + TIME_LIMIT
    start_new_thread( runRadar, ( startTime, endTime ) )
    start_new_thread( runIR, ( startTime, endTime ) )
    camera = picamera.PiCamera()
    camera.resolution = (640,480)
    camera.start_preview()
    while time.time() < endTime:
        t = time.time() - startTime
        t = int( round( t*1000, 3 ) ) #time in miliseconds
        camera.capture(VIDEO_DIR+"/im%04d.png" %t, resize = (160, 120), use_video_port=True)
    time.sleep(0.5)
    camera.stop_preview()
    time.sleep(1)
    camera.close()
    waiting4finish()


if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print( __doc__)
        sys.exit()
    label = sys.argv[1]
    if len(sys.argv) > 2:
        TIME_LIMIT = int(sys.argv[2])
        print( "New TIME LIMIT ", TIME_LIMIT)
    notes = open("logs/notes_"+label+".txt", "w")
    notes.write(label+"\r\n")
    notes.write("TIME_Limit %0d\r\n" %TIME_LIMIT)
    notes.write("RADAR_LOG "+RADAR_LOG+"\r\n")
    notes.write("IR_LOG "+IR_LOG+"\r\n")
    notes.write("VIDEO_DIR "+VIDEO_DIR+"\r\n")
    notes.close()
    deerDetect()
