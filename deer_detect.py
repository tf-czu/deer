#!/usr/bin/python3
"""
    usage:
         sudo python3 deer_detect.py label
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import datetime
import time
import os
import numpy as np
import cv2
import pyaudio
from threading import Thread
#import picamera


CHUNK = 800 #1024
FORMAT = pyaudio.paInt16 #pyaudio.paInt16
CHANNELS = 1
RATE = 4000 #44100 96000
RADAR_LOG = datetime.datetime.now().strftime("logs/radar_%y%m%d_%H%M%S") + ".log"
IR_LOG = datetime.datetime.now().strftime("logs/ir_%y%m%d_%H%M%S") + ".log"
VIDEO_DIR = datetime.datetime.now().strftime("logs/video_%y%m%d_%H%M%S")
os.mkdir(VIDEO_DIR)
TIME_LIMIT = 10

g_radarEnd = True
g_irEnd = True


def findMax(y):
    yd = np.diff(y)
    ydd = np.diff(yd)
    ex = yd[:-1] * yd[1:]
    y_max = (ex<=0) & (ydd<=0)
    
    return y[1:-1][y_max], np.nonzero(y_max)[0] + 1


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
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input_device_index=None, input=True, frames_per_buffer=CHUNK )
    print( "* recording radar data")
    frames = []
    tId = []
    radarLog = open( RADAR_LOG, "w" )
    freq = np.fft.fftfreq(800, d= 1/RATE)
    freq = freq[:50]
    g_radarEnd = False
    while time.time() < endTime:
        try:
            data = stream.read(CHUNK, exception_on_overflow = False)
            data = np.fromstring( data, dtype=np.int16)
            fft = np.abs( np.fft.rfft( data ) / data.size )[:50]
            
            am_max, args = findMax(fft)
            arg_max = np.argmax(am_max)
            
            if len(am_max) > 0:
                #sufficient frequency
                freq_main = freq[ args[ arg_max ] ]
                sufficient_freq = freq_main > 30 and freq_main < 300
                
                #sufficient amplitude
                sufficient_am = am_max[ arg_max ] > 500
                
                #amplitudes relations
                numMax = len(am_max)
                if numMax == 1:
                    amp_relations = True
                else:
                    numMax = min(numMax, 4)
                    am_max_sort = np.sort(am_max)[::-1]
                    ratio = am_max_sort[0] / np.sum(am_max_sort[1:numMax])
                    amp_relations = ratio > 2
                    
                deer = sufficient_freq and sufficient_am and amp_relations
                print(freq_main, am_max[ arg_max ], ratio)
                print(sufficient_freq, sufficient_am, amp_relations)
                
            else:
                deer = False
                
            if deer:
                print("-----DEER-----")
            
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
            radarLog.write( str( list( frame ) ) )
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
    fifo = open('/var/run/mlx9062x.sock', 'rb')#, encoding ='utf-8')
    time.sleep(0.1)
    irLog = open(IR_LOG, "w")
    irData = []
    tId = []
    g_irEnd = True
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


def runCamera( startTime, endTime ):
    try:
        camera = picamera.PiCamera()
        camera.resolution = (640,480)
        #time.sleep(0.1)
        camera.start_preview()
        time.sleep(2)
        while time.time() < endTime:
            t = time.time() - startTime
            t = int( round( t*1000, 3 ) ) #time in miliseconds
            camera.capture(VIDEO_DIR+"/im%04d.png" %t, use_video_port=True)
            
        time.sleep(0.5)
        camera.stop_preview()
        time.sleep(1)
        camera.close()
        print('camera.close')
        
    except:
        print('camera fall')
        

def runCamera2(startTime, endTime):
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    while time.time() < endTime:
        t = time.time() - startTime
        t = int( round( t*1000, 3 ) )
        ret, frame = cap.read()
        cv2.imwrite(VIDEO_DIR+"/im%04d.png" %t, frame)
        time.sleep(0.1)
    cap.release()


def deerDetect( ):
    startTime = time.time()
    endTime = startTime + TIME_LIMIT
    Thread( target=runRadar, args=(startTime, endTime ) ).start()
    Thread( target=runIR, args=(startTime, endTime ) ).start()
    Thread( target=runCamera2, args=(startTime, endTime ) ).start()
    
    time.sleep(TIME_LIMIT + 3)
    
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
