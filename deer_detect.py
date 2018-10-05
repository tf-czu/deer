#!/usr/bin/python3
"""
    usage:
        sudo python3 deer_detect.py label
    or:
        python3 deer_detect.py log <noteLog>
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
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import scipy.stats as stats
#import picamera


CHUNK = 400 #800 #1024
FORMAT = pyaudio.paInt16 #pyaudio.paInt16
CHANNELS = 1
RATE = 4000 #44100 96000
TIME_LIMIT = 10
T_DIFF = 3
AREA_LIMIT = 4

mser = cv2.MSER_create( _delta = 5, _min_area=3, _max_area=16)
ZERO_IR = np.array([30230, 30112, 30205, 30244, 30145, 30206, 30160, 30400, 30081, 30203, 30124, 30325, 30198, 30335, 30325, 30260, 
           30318, 30182, 30250, 30327, 30231, 30263, 30298, 30300, 30356, 30376, 30342, 30315, 30346, 30334, 30238, 30500, 
           30381, 30511, 30363, 30413, 30353, 30356, 30435, 30439, 30343, 30295, 30468, 30411, 30541, 30542, 30401, 30656, 
           30541, 30567, 30440, 30656, 30544, 30503, 30448, 30778, 30821, 30605, 30648, 30727, 30466, 30751, 30537, 30631])
ZERO_IR = np.reshape( ZERO_IR, (16,4) ).T/100 - 273.15
random_background = np.random.rand(6,18)*2 - 1

g_radarEnd = True
g_irEnd = True
g_deer_R = None
g_deer_IR = None


def evaluateLog(notesFile):
    path, fileN = os.path.split(notesFile)
    notes = open(notesFile)
    for line in notes:
        line = line.split()
        if line[0] == "RADAR_LOG":
            rLogFile = os.path.join(path, line[1])
            print( rLogFile)
        elif line[0] == "IR_LOG":
            irFile = os.path.join(path, line[1])
            print( irFile)
        elif line[0] == "VIDEO_DIR":
            videoDir = os.path.join(path, line[1])
            print( videoDir)
    return rLogFile, irFile, videoDir


def findMax(y):
    yd = np.diff(y)
    ydd = np.diff(yd)
    ex = yd[:-1] * yd[1:]
    y_max = (ex<=0) & (ydd<=0)
    
    return y[1:-1][y_max], np.nonzero(y_max)[0] + 1


def radarProcessing(data, timeId = None, rDir = None, im = None):
    freq = np.fft.fftfreq(CHUNK, d= 1/RATE)
    #print(len(freq))
    freq = freq[:100]
    
    fft = np.abs( np.fft.rfft( data ) / data.size )[:100]
    am_max, args = findMax(fft)
    arg_max = np.argmax(am_max)
    
    if len(am_max) > 0:
        #sufficient frequency
        freq_main = freq[ args[ arg_max ] ]
        sufficient_freq = freq_main > 100 and freq_main < 300
        
        #sufficient amplitude
        sufficient_am = am_max[ arg_max ] > 3000
        
        """
        #kurtosis
        fft_section = fft[10:30] #frequencies 100 - 300 Hz
        kurt = stats.kurtosis(fft_section)
        amp_relations = kurt > 1
        """
        #amplitudes relations
        numMax = len(am_max)
        if numMax == 1:
            amp_relations = True
        else:
            numMax = min(numMax, 4)
            am_max_sort = np.sort(am_max)[::-1]
            ratio = am_max_sort[0] / np.sum(am_max_sort[1:numMax])
            amp_relations = ratio > 0.5
            
        deer = sufficient_freq and sufficient_am and amp_relations
        #print(freq_main, am_max[ arg_max ], ratio)
        #print(sufficient_freq, sufficient_am, amp_relations)
        
    else:
        deer = False
    
    if timeId is not None and deer:
        fig = plt.figure(figsize=(5,5))
        fig.subplots_adjust(left = 0.15, bottom=0.15, hspace = 0.25)
        ax1 =fig.add_subplot(211)
        #ax1.plot(freq[:50], fft, "k-")
        #ax1.plot(freq[:50][args], am_max, "ro")
        ax1.plot(freq, fft, "k-")
        ax1.plot(freq[args], am_max, "ro")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude (-)")
        ax1.set_ylim(0,5000)
        if deer:
            ax1.text(300, 4500, "Deer detected", fontsize = 14, color = "red")
        #ax1.text(300, 4000, str(kurt), fontsize = 14, color = "red")
        
        if im is not None:
            ax2 = fig.add_subplot(212)
            ax2.imshow(im)
            ax2.axis("off")
        
        figName = os.path.join(rDir, "fft_%05d" %timeId)
        plt.savefig(figName, dpi = 100)
        plt.close()
    
    return deer


def runRadar( startTime, endTime ):
    global g_radarEnd, g_deer_R
    g_radarEnd = False
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input_device_index=None, input=True, frames_per_buffer=CHUNK )
    print( "* recording radar data")
    frames = []
    tId = []
    radarLog = open( "logs/"+RADAR_LOG, "w" )
    while time.time() < endTime:
        try:
            data = stream.read(CHUNK, exception_on_overflow = False)
            data = np.fromstring( data, dtype=np.int16)
            deer = radarProcessing(data)
            if deer:
                print("-----DEER_R-----")
                g_deer_R = time.time()
        except:
            data = "0"
            print("no data")
        tId.append(time.time() - startTime)
        frames.append(data)
        #continue
    
    print("* done recording")
    for ii, frame in enumerate( frames ):
        radarLog.write(str(tId[ii])+"&")
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


def irProcessing(data, ii = None, irDir = None, im = None):
    deer = False
    t_min = 10
    t_max = 40
    ir_frame = np.reshape( data, (16,4) ).T/100 - 273.15
    ir_frame2 = ir_frame - ZERO_IR + ZERO_IR.mean()
    mean_t = np.mean(ir_frame2)
    #background = np.ones((6,18))*mean_t - random_background
    #background[1:5, 1:17] = ir_frame2
    #ir_frame2 = background
    ir_frame_int =  ( ir_frame2 - t_min ) * 255 / ( t_max - t_min )
    ir_frame_int[ ir_frame_int < 0 ] = 0
    ir_frame_int[ ir_frame_int > 255 ] = 255
    ir_frame_int = ir_frame_int.astype(np.uint8)
    ir_frame_int = cv2.resize(ir_frame_int,(32, 8), interpolation = cv2.INTER_CUBIC)
    
    """
    regions, bboxes = mser.detectRegions(ir_frame_int)
    if ii is not None:
        print(ii, len(regions), regions, bboxes)
    targets = []
    for region in regions:
        t_reg = np.mean(ir_frame2[ region[:,1], region[:,0] ] )
        if ii is not None:
            print(mean_t, t_reg, t_reg - mean_t)
        if t_reg - mean_t > 2.5:
            targets.append(region)
    #print(targets)
    if len(targets) > 0:
        deer = True
    """
    mode, ___ = stats.mode(np.round(ir_frame2), axis=None)
    treshValue = int(( mode + T_DIFF - t_min ) * 255 / ( t_max - t_min ))
    ret, binaryImg = cv2.threshold( ir_frame_int, treshValue, 255,cv2.THRESH_BINARY)
    ___, contours, ___ = cv2.findContours(binaryImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    a_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        a_list.append(area)
        if area > AREA_LIMIT:
            deer = True
        
    print(a_list)
    
    if ii is not None:
        print(ii, mode)
        plt.hist( ir_frame2.ravel(), 256,[0,256])
        histName = os.path.join(irDir, "IRH_%05d" %ii)
        plt.savefig(histName, dpi=100)
        plt.close()
        
        fig = plt.figure(figsize=(6,5))
        fig.subplots_adjust(left= 0.05, right=0.95, hspace=0.4, wspace = 0.22, top=0.95)
        ax1 = fig.add_subplot(411)
        irBox = ax1.pcolormesh(ir_frame, vmin = t_min, vmax = t_max ) #cmap = 'Greys'
        ax1.axis("off")
        cb = fig.colorbar(irBox, ax=ax1)
        cb.ax.tick_params(labelsize=4)
        
        ax2 = fig.add_subplot(412)
        #ir_frame_int[0,0] = 0
        irBox = ax2.pcolormesh(ir_frame_int, vmin = 0, vmax = 255 ) #cmap = 'Greys'
        ax2.axis("off")
        cb = fig.colorbar(irBox, ax=ax2)
        cb.ax.tick_params(labelsize=4)
        
        ax3 = fig.add_subplot(413)
        ir_frame_targets = ir_frame2.copy()
        #for targ in targets:
        #    ir_frame_targets[ targ[:,1], targ[:,0] ] = 45
            
        irBox = ax3.pcolormesh(binaryImg, vmin = t_min, vmax = 40 ) #cmap = 'Greys'
        ax3.axis("off")
        cb = fig.colorbar(irBox, ax=ax3)
        cb.ax.tick_params(labelsize=4)
        
        if im is not None:
            ax4 = fig.add_subplot(414)
            ax4.imshow(im)
            ax4.axis("off")
        
        figName = os.path.join(irDir, "IR_%05d" %ii)
        plt.savefig(figName, dpi=100)
        plt.close()
    
    return deer


def runIR(startTime, endTime ):
    global g_irEnd, g_deer_IR
    fifo = open('/var/run/mlx9062x.sock', 'rb')#, encoding ='utf-8')
    time.sleep(0.1)
    irLog = open("logs/"+IR_LOG, "w")
    irData = []
    tId = []
    g_irEnd = False
    while time.time() < endTime:
        ir_raw = fifo.read()
        if len(ir_raw) < 128:
            continue
        ir_last = ir_raw[-128:]
        tId.append(time.time() - startTime)
        # go all numpy on it
        ir = np.frombuffer(ir_last, np.uint16)
        deer = irProcessing(ir)
        if deer:
            print("-----DEER_IR-----")
            g_deer_IR = time.time()
        irData.append(list(ir))
    
    fifo.close()
    for ii, irD in enumerate( irData ):
        irLog.write(str(tId[ii])+"&")
        irLog.write(str(irD))
        irLog.write( "\r\n" )
    irLog.close()
    
    g_irEnd = True


def runCamera( startTime, endTime):
    try:
        camera = picamera.PiCamera()
        camera.resolution = (640,480)
        #time.sleep(0.1)
        camera.start_preview()
        time.sleep(2)
        while time.time() < endTime:
            t = time.time() - startTime
            t = int( round( t*1000, 3 ) ) #time in miliseconds
            camera.capture("logs/"+VIDEO_DIR+"/im%04d.png" %t, use_video_port=True)
            
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
        cv2.imwrite("logs/"+VIDEO_DIR+"/im%04d.png" %t, frame)
        time.sleep(0.1)
    cap.release()


def evalLog(log):
    rLogFile, irFile, videoDir = evaluateLog(log)
    
    imList_FN = sorted( os.listdir( videoDir ) )
    imList = []
    t_im = []
    for imF in imList_FN:
        im_name, endswitch = imF.split(".")
        if endswitch == "png":
            imList.append( cv2.imread( os.path.join(videoDir, imF) ) )
            t_im.append( int(im_name[2:])/1000 )
    t_im = np.array(t_im)
    
    print( "Radar data")
    dirName, fileN = os.path.split(log)
    rDir = os.path.join(dirName, fileN.split(".")[0]+"_radar")
    if not os.path.isdir(rDir):
        os.mkdir(rDir)
    
    rLog = open(rLogFile)
    timeR = []
    dataR = []
    for line in rLog:
        tId, data = line.split("&")
        timeR.append(float(tId))
        dataR.append( eval(data) )
    
    dataR = np.array(dataR)
        
    for ii, data in enumerate(dataR):
        #print(ii)
        #print("-----------")
        t_r = timeR[ii]
        argv_t_im = np.argmin( np.abs( t_im - t_r) )
        im = imList[argv_t_im]
        deer = radarProcessing(data, timeId = ii, rDir = rDir, im = im)
        if deer:
            print(ii, "DEER")
    #sys.exit()
    print( "IR data")
    irF = open(irFile)
    time_ir = []
    irData = []
    for line in irF:
        #print(line)
        t, ir = line.split("&")
        time_ir.append(float(t))
        irData.append(eval(ir))
    irData = np.array(irData)
    print(irData.shape)
    irDir = os.path.join(dirName, fileN.split(".")[0]+"_ir")
    if not os.path.isdir(irDir):
        os.mkdir(irDir)
    
    for ii, irD in enumerate(irData):
        #print(ii)
        #print("-----------")
        t_ir = time_ir[ii]
        argv_t_im = np.argmin( np.abs( t_im - t_ir) )
        im = imList[argv_t_im]
        deer = irProcessing(irD, ii = ii, irDir = irDir, im = im)
        if deer:
            print(ii, "DEER")


def deerDetect(  ):
    global g_radarEnd, g_irEnd, g_deer_R, g_deer_IR
    deer_log = open("logs/"+DEER_LOG, "w")
    startTime = time.time()
    endTime = startTime + TIME_LIMIT
    Thread( target=runRadar, args=(startTime, endTime ) ).start()
    Thread( target=runIR, args=(startTime, endTime ) ).start()
    Thread( target=runCamera2, args=(startTime, endTime ) ).start()
    
    time.sleep(1)
    while True:
        if g_deer_R and g_deer_IR:
            if abs(g_deer_R - g_deer_IR) < 0.2:
                if time.time() - ( g_deer_R + g_deer_IR )/2.0 < 0.2:
                    print("!!--------DEER--------!!")
                    deer_log.write(str( [g_deer_R - startTime, g_deer_IR - startTime] ) + "\r\n" )
        
        if g_radarEnd and g_irEnd:
            break
        time.sleep(0.05)
    deer_log.close()
    time.sleep(1)


if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print( __doc__)
        sys.exit()
    label = sys.argv[1]
    if label == "log":
        TIME_LIMIT = 0
        evalLog(log = sys.argv[2] )
    else:
        if len(sys.argv) > 2:
            TIME_LIMIT = int(sys.argv[2])
            print( "New TIME LIMIT ", TIME_LIMIT)
        RADAR_LOG = datetime.datetime.now().strftime("radar_%y%m%d_%H%M%S") + ".log"
        IR_LOG = datetime.datetime.now().strftime("ir_%y%m%d_%H%M%S") + ".log"
        VIDEO_DIR = datetime.datetime.now().strftime("video_%y%m%d_%H%M%S")
        os.mkdir("logs/"+VIDEO_DIR)
        DEER_LOG = datetime.datetime.now().strftime("deer_%y%m%d_%H%M%S") + ".log"
        notes = open("logs/notes_"+label+".txt", "w")
        notes.write(label+"\r\n")
        notes.write("TIME_Limit %0d\r\n" %TIME_LIMIT)
        notes.write("RADAR_LOG "+RADAR_LOG+"\r\n")
        notes.write("IR_LOG "+IR_LOG+"\r\n")
        notes.write("VIDEO_DIR "+VIDEO_DIR+"\r\n")
        notes.write("DEER_LOG "+DEER_LOG+"\r\n")
        notes.close()
        deerDetect()
