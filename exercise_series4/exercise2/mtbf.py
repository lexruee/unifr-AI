'''
  Copyright (c) 2012, Markus Weber, Marcus Eichenberger-Liwicki. All rights reserved.
  
  Redistribution and use in source and binary forms, with or without modification, are
  permitted provided that the following conditions are met:
  
  Redistributions of source code must retain the above copyright notice, this list of
  conditions and the following disclaimer. Redistributions in binary form must reproduce the
  above copyright notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
  
  Neither the name of the author nor the names of its contributors may be used to endorse or
  promote products derived from this software without specific prior written permission.
  
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import struct as st
import scipy.io as matio
import numpy as np
import logging
import time
import os
import math
import tempfile


''' Logger '''
log = logging.getLogger("aai.dataset.mtbf")
''' Sample rate of the motion sensor. '''
SENSOR_SAMPLERATE = 100.0     # Hz
DEFAULT_SAMPLING_RATE = 20    # Hz
MTBF_BYTE_MINSIZE = 7 + 8 + 8 + 1 # Byte


# Body part labels
BODY_LABEL = {  'torso' : 'T',
                'leftarm:upper' : 'LA:ES',
                'leftarm:lower' : 'LA:EW',
                'rightarm:upper' : 'RA:ES',
                'rightarm:lower' : 'RA:EW',
                'pelvis' : 'P',
                'leftleg:upper' : 'LL:HK',
                'leftleg:lower' : 'LL:KF',
                'rightleg:upper' : 'RL:HK',
                'rightleg:lower' : 'RL:KF'
                }

def crc8(data):
    #copy-paste from mtbf.cpp
    chksum = 0xFF
    for i in xrange(len(data)) :
        chksum ^= ord(data[i])
    return chksum

def testBit(int_type, offset):
    '''
        Testing the bit for on a position.
        :Parameters:
            int_type : an integer value
            offset : position
    '''
    mask = 1 << offset
    return bool(int_type & mask)

def setBit(int_type, offset):
    '''
        Sets a single bit.
        :Parameters:
            int_type : an integer value
            offset : position
    '''
    mask = 1 << offset
    return(int_type | mask)

def configure3DOF(data):
    '''
        Read data for a 3DOF from MTBF.
        :Parameters:
            data  :  data segment set to the correct starting position
        :Returns:
            euler angle [x, y, z]
    '''
    return (st.unpack('f', data[0:4])[0],
            st.unpack('f', data[4:8])[0],
            st.unpack('f', data[8:12])[0])

def configure2DOF(data):
    '''
        Read data for a 2DOF from MTBF.
        :Parameters:
            data  :  data segment set to the correct starting position
        :Returns:
            euler angle [x, z]
    '''
    return (st.unpack('f', data[0:4])[0], 
            0.0,
            st.unpack('f', data[4:8])[0])

class MTBFData(object):
    '''
     Motion Tracker Binary Format Data.
  
     Message structure:
     ------------------
     PREAMBLE   TIME                  FLAGS                  DATA      CHECKSUM
     (7 byte)   (64 bit int, 8 byte)  (64 bit int, 8 byte)             (1 byte)
 
     |0     6 |7                  14|15                   22|23 .....
 
     PREAMBLE:
     ---------
     Every message starts with the preamble: 'M', 'T', 'B', 'F', '0', '.', '4'.
 
     TIME:
     -----
     Every message contains an absolute time stamp given based on the POSIX/Unix time format.
     In order to be able to represent sub-second precision, the 48 most significant bits are
     used for the POSIX time part, and the 16 least significant bits are used for the fractional part.
 
     FLAGS:
     ------
     The flags define the width and content of the DATA field. Since a 64 bit integer is used, 64 different
     data types can be defined. In order to keep the format extensible, one or more of these could be defined
     to host additional content specifications to be hidden as data. The content is stored in the DATA field.
 
     DATA:
     -----
     The data field. Depending on the flags the data field contains some values. The list of value for each
     flag is sorted depending on the order of the flags.
 
     CHECKSUM:
     ---------
     This field is used for communication error-detection.
 
     Bit index    Size [byte]   Data content    Data identifier   Data description
     -----------------------------------------------------------------------------
      0           12            3 floats        TORSO             Yaw, pitch, roll angles of torso in rad
      1           12            3 floats        UR                Yaw, pitch, roll angles of upper right shoulder in rad
      2            8            2 floats        LR                Yaw, roll angles of right elbow in rad
      3           12            3 floats        UL                Yaw, pitch, roll angles of upper left shoulder in rad
      4            8            2 floats        LL                Yaw, roll angles of left elbow in rad
      5            1            1 byte          ACTIVITY          Activity id (e.g. walking, running, cycling, ...)
      6            1            1 byte          INTENSITY         Intensity of the activity. Currently, the range of one
                                                                  byte is divided into three parts equal to low, medium,
                                                                  high intensity.
      7            1            1 byte          INFO              Used to send simple information. Bit 0 reserved for
                                                                  repetition completed.
      8            2            1 uint16        WARNING           Set of predefined warnings, one per bit
      9            2            1 uint16        MODE              Number indicating what mode has been started (0xFFFF is
                                                                  reserved to indicate failure to execute command.)
                                                                  This is used to keep the information, which
                                                                  exercises were executed in order to be able
                                                                  to reconstruct the training session during replay.
     10            1            1 byte          SENSOR            Defines the number of available sensors.
                                                                  See sensor definition for further details.
     11           12            3 floats        PELVIS            Yaw, pitch, roll angles of pelvis in rad
     12           12            3 floats        URL               Yaw, pitch, roll angles of right hip (upper right leg)
                                                                  in rad
     13            8            2 floats        LRL               Yaw, roll angles of right knee (lower right leg) in rad
                                                                  (roll unused right now)
     14           12            3 floats        ULL               Yaw, pitch, roll angles of left hip (upper left left)
                                                                  in rad
     15            8            2 floats        LLL               Yaw, roll angles of left knee (lower left leg) in rad
     16            1            1 byte          HEARTRATE         Heart rate in bpm.
 
     ACTIVITY (1 byte)
     -----------------
     The type of activity from general/aerobic activity monitoring.
 
     Bit:             Activity
     7 6 5 4 3 2 1 0
     ----------------
     0 0 0 0 0 0 0 0  OTHER (draw at actual intensity)
     0 0 0 0 0 0 0 1  LYING
     0 0 0 0 0 0 1 0  SITTING / STANDING
     0 0 0 0 0 0 1 1  WALKING
     0 0 0 0 0 1 0 0  RUNNING
     0 0 0 0 0 1 0 1  CYCLING
     0 0 0 0 0 1 1 0  NORDIC WALKING
 
     INTENSITY (1 byte)
     ------------------
     The intensity from general/aerobic activity monitoring.
     0x00                NONE
     0x01-0x55 (0x2B)    LOW
     0x56-0xAA (0x80)    MEDIUM
     0xAB-0xFF (0xD5)    HIGH
 
     INFO (1 byte)
     -------------
     Bit:             Info
     7 6 5 4 3 2 1 0
     ----------------
     0 0 0 0 0 0 0 0  NOTHING
     0 0 0 0 0 0 0 1  REPETITION_COMPLETED/DONE
                      (completed repetition for current exercise/calibration step done)
 
     WARNING (1 uint 16)
     -------------------
     This will be updated with the nature of the warnings during strength exercise training based on the algorithm of
     UTC: in terms of range of motion; posture; velocity; smoothness. The user will be warned when one of these
     parameters deviates abnormally from the movement of reference. Moreover, a warning for safe heart rate exceeded
     will be added.
 
     Bits:                                               Warning
     15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
     -----------------------------------------------
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0     NOTHING
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1     SHOULDER
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0     FLEXOR
      0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0     EXTENSOR
      0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0     RANGE OF MOTION
      0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0     POSTURE
      0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0     VELOCITY
      0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0     SMOOTHNESS
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 
     MODE (1 uint 16)
     ----------------d
 
     ID for the supported exercises. If reference motions are recorded in a generic way, IDs will have to be
     assigned generically.
 
     Bits:                                               Mode
     15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
     -----------------------------------------------
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0     Sensor check
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1     Calibration
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0
      0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0
      0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0
      0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0
      0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0
      0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 
     SENSOR (1 byte):
     ----------------
     ID of the IMUs according to their positioning.
 
     Bit:             Sensor
     7 6 5 4 3 2 1 0
     ----------------
     0 0 0 0 0 0 0 0  NOTHING
     0 0 0 0 0 0 0 1  TORSO
     0 0 0 0 0 0 1 0  LEFT ARM
     0 0 0 0 0 1 0 0  RIGHT ARM
     0 0 0 0 1 0 0 0  PELVIS
     0 0 0 1 0 0 0 0  LEFT LEG
     0 0 1 0 0 0 0 0  RIGHT LEG

    '''
    def __init__(self, protocol_version, packet):
        '''
        Constructor
        '''
        self.flags = 0
        if packet is not None : 
            self._parse_(packet)

    def get_timestamp(self):
        return self.__timestamp

    def set_timestamp(self, value):
        self.__timestamp = value

    def del_timestamp(self):
        del self.__timestamp

    def get_heartrate(self):
        return self.__heartrate

    def set_heartrate(self, value):
        self.__heartrate = value
        self.flags = setBit(self.flags, 16)

    def del_heartrate(self):
        del self.__heartrate

    def get_mode(self):
        return self.__mode

    def set_mode(self, value):
        self.__mode = value

    def del_mode(self):
        del self.__mode

    def get_pelvis(self):
        return self.__pelvis

    def set_pelvis(self, value):
        self.__pelvis = value

    def del_pelvis(self):
        del self.__pelvis

    def get_torso(self):
        return self.__torso

    def get_ur(self):
        return self.__ur

    def get_lr(self):
        return self.__lr

    def get_ul(self):
        return self.__ul

    def get_ll(self):
        return self.__ll

    def get_url(self):
        return self.__url

    def get_lrl(self):
        return self.__lrl

    def get_ull(self):
        return self.__ull

    def get_lll(self):
        return self.__lll

    def set_torso(self, value):
        self.__torso = value
        self.flags = setBit(self.flags, 0)

    def set_ur(self, value):
        self.__ur = value
        self.flags = setBit(self.flags, 1)

    def set_lr(self, value):
        self.__lr = value
        self.flags = setBit(self.flags, 2)

    def set_ul(self, value):
        self.__ul = value
        self.flags = setBit(self.flags, 3)

    def set_ll(self, value):
        self.__ll = value
        self.flags = setBit(self.flags, 4)

    def set_url(self, value):
        self.__url = value
        self.flags = setBit(self.flags, 12)

    def set_lrl(self, value):
        self.__lrl = value
        self.flags = setBit(self.flags, 13)

    def set_ull(self, value):
        self.__ull = value
        self.flags = setBit(self.flags, 14)

    def set_lll(self, value):
        self.__lll = value
        self.flags = setBit(self.flags, 15)

    def del_torso(self):
        del self.__torso

    def del_ur(self):
        del self.__ur

    def del_lr(self):
        del self.__lr

    def del_ul(self):
        del self.__ul

    def del_ll(self):
        del self.__ll

    def del_url(self):
        del self.__url

    def del_lrl(self):
        del self.__lrl

    def del_ull(self):
        del self.__ull

    def del_lll(self):
        del self.__lll

    def __str__(self, *args, **kwargs):
        return "{} [timestamp:={}] [torso]:{} [larm]:{} [rarm]:{} [pelvis]:{} [lleg]:{} [rleg]:{}"\
                .format(self.get_preamble(),
                        self.get_timestamp(),
                        self.has_torso(),
                        self.has_left_arm(),
                        self.has_right_arm(),
                        self.has_pelvis(),
                        self.has_left_leg(),
                        self.has_right_leg())
    def __getitem__(self, key) :
        if isinstance(key, str) :
            if key == 'torso' : return self.get_torso()
            if key == 'pelvis' : return self.get_pelvis()
            if key == 'leftarm' : return (self.get_ul(), self.get_ll())
            if key == 'rightarm' : return (self.get_ur(), self.get_lr())
            if key == 'leftleg' : return (self.get_url(), self.get_lrl())
            if key == 'rightleg' : return (self.get_ull(), self.get_lll())

    def get_preamble(self):
        return self.__preamble

    def get_warning_flag(self):
        return self.__warningFlag

    def get_activity_flag(self):
        return self.__activityFlag

    def get_intensity_flag(self):
        return self.__intensityFlag

    def get_info(self):
        return self.__info

    def get_sensor(self):
        return self.__sensor

    def get_data(self):
        return self.__data

    def set_preamble(self, value):
        self.__preamble = value

    def set_warning_flag(self, value):
        if value is None : return
        assert(type(value) == int)
        self.__warningFlag = value
        self.flags = setBit(self.flags, 8)

    def set_activity_flag(self, value):
        self.__activityFlag = value
        self.flags = setBit(self.flags, 5)

    def set_intensity_flag(self, value):
        self.__intensityFlag = value
        self.flags = setBit(self.flags, 6)

    def set_info(self, value):
        if value is None : return
        assert(value == 0 or value == 1)
        self.__info = value
        self.flags = setBit(self.flags, 7)
        
    def set_sensor(self, value):
        self.__sensor = value

    def set_data(self, value):
        self.__data = value

    def del_preamble(self):
        del self.__preamble

    def del_warning_flag(self):
        del self.__warningFlag

    def del_activity_flag(self):
        del self.__activityFlag

    def del_intensity_flag(self):
        del self.__intensityFlag

    def del_info(self):
        del self.__info

    def del_sensor(self):
        del self.__sensor

    def del_data(self):
        del self.__data
        
    def _parse_(self, packet) :
        assert(packet[:4] == "MTBF")
        '''
          PREAMBLE   TIME                  FLAGS                  DATA      CHECKSUM
         (7 byte)   (64 bit int, 8 byte)  (64 bit int, 8 byte)             (1 byte)
 
         |0     6 |7                  14|15                   22|23 .....
        '''
        # little endian
        int_format = '<Q' 
        self.set_preamble(packet[:7])
        ts = st.unpack(int_format, packet[7:15])[0]
        '''
            uint64_t s = time >> 16;
            uint16_t ms = static_cast<uint16_t>((time&0xFFFF) % 1000);
           _time = static_cast<double>(s) + static_cast<double>(ms)/1000.;
           TIME:
           -----
             Every message contains an absolute time stamp given based on the POSIX/Unix time format.
             In order to be able to represent sub-second precision, the 48 most significant bits are
             used for the POSIX time part, and the 16 least significant bits are used for the fractional part.
        '''
        s = ts >> 16
        ms = (ts & 0xFFFF) % 1000
        self.timestamp = float(s) + float(ms) / 1000
        self.flags = st.unpack(int_format, packet[15:23])[0]
        data = packet[23:]
        ipos = 0
        if self.has_torso():
            self.torso = configure3DOF(data[ipos:])
            ipos += 12
        if self.has_ur():
            self.ur = configure3DOF(data[ipos:])
            ipos += 12
        if self.has_lr():
            self.lr = configure2DOF(data[ipos:])
            ipos += 8
        if self.has_ul():
            self.ul = configure3DOF(data[ipos:])
            ipos += 12
        if self.has_ll():
            self.ll = configure2DOF(data[ipos:])
            ipos += 8
        if self.has_activity():
            self.activityFlag = st.unpack('b', data[ipos])[0]
            ipos += 1
        if self.has_intensity():
            self.intensityFlag = st.unpack('b', data[ipos])[0]
            ipos += 1
        if self.has_info():
            self.info = st.unpack('b', data[ipos])[0]
            ipos += 1
        if self.has_warning():
            self.warningFlag = st.unpack('H', data[ipos:ipos + 2])[0]
            ipos += 2
        if self.has_mode():
            self.mode = st.unpack('H', data[ipos:ipos + 2])[0]
            ipos += 2
        if self.has_sensor():
            self.sensor = st.unpack('b', data[ipos])[0]
            ipos += 1
        if self.has_pelvis():
            self.pelvis = configure3DOF(data[ipos:])
            ipos += 12
        if self.has_url():
            self.url = configure3DOF(data[ipos:])
            ipos += 12
        if self.has_lrl():
            self.lrl = configure2DOF(data[ipos:])
            ipos += 8
        if self.has_ull():
            self.ull = configure3DOF(data[ipos:])
            ipos += 12
        if self.has_lll():
            self.lll = configure2DOF(data[ipos:])
            ipos += 8
        if self.has_heartrate():
            self.heartrate = st.unpack('b', data[ipos])[0]
            ipos += 1
        
        crc = st.unpack('B', data[ipos])[0]
        assert crc == crc8(packet[:-1])
    
    def encode_binary(self):
        '''
            Encodes the MTBF data packet in its binary representation.
            
            :Returns:
                Binary representation of MTBF data
        '''
        int_format_8b = '<Q'
        float_format = '<f'
        tor = int(self.has_torso())
        ura = int(self.has_right_arm())
        lra = int(self.has_right_arm())
        ula = int(self.has_left_arm())
        lla = int(self.has_left_arm())
        act = int(self.has_activity())
        its = int(self.has_intensity())
        ifo = int(self.has_info())
        wrn = int(self.has_warning())
        mod = int(self.has_mode())
        sen = int(self.has_sensor())
        pel = int(self.has_pelvis())
        url = int(self.has_right_leg())
        lrl = int(self.has_right_leg())
        ull = int(self.has_left_leg())
        lll = int(self.has_left_leg())
        hrt = int(self.has_heartrate())
        flagbin = "".join([str(hrt), str(lll), str(ull), str(lrl), str(url), str(pel), 
                           str(sen), str(mod), str(wrn), str(ifo), str(its),str(act), 
                           str(lla), str(ula), str(lra), str(ura), str(tor)])
        flag = st.pack(int_format_8b,int(flagbin,2))
        preamble = "MTBF0.4".encode("ascii")
        '''
            unsigned long s = static_cast<unsigned long>(std::floor(time))+_time;
            unsigned short ms = static_cast<unsigned short>(std::fmod(time, 1.)*1000);
            s <<= 16;
            s |= ms % 1000;
            _buf.append(reinterpret_cast<char*>(&s), sizeof(s));
            _buf.append(sizeof(uint64_t), '\0');
        '''
        s = int(math.floor(self.timestamp))
        ms = int(math.fmod(self.timestamp, 1e100)*1000)
        s = s << 16
        s |= ms % 1000
        tstamp = st.pack(int_format_8b,s)
        data = ''
        if tor > 0 :
            data += st.pack(float_format,self.torso[0])
            data += st.pack(float_format,self.torso[1])
            data += st.pack(float_format,self.torso[2])
        if ura > 0 :
            data += st.pack(float_format,self.ur[0])
            data += st.pack(float_format,self.ur[1])
            data += st.pack(float_format,self.ur[2])
        if lra > 0 :
            data += st.pack(float_format,self.lr[0])
            data += st.pack(float_format,self.lr[2])
        if ula > 0 :
            data += st.pack(float_format,self.ul[0])
            data += st.pack(float_format,self.ul[1])
            data += st.pack(float_format,self.ul[2])
        if lla > 0 :
            data += st.pack(float_format,self.ll[0])
            data += st.pack(float_format,self.ll[2])
        if self.has_activity():
            data += st.pack('b', self.activityFlag)
        if self.has_intensity():
            data += st.pack('b', self.intensityFlag)
        if self.has_info():
            data += st.pack('b', self.info)
        if self.has_warning():
            data += st.pack('H', self.warningFlag)
        if self.has_mode():
            data += st.pack('H', self.mode)
        if self.has_sensor():
            data += st.pack('b', self.info)
        if pel > 0 :
            data += st.pack(float_format,self.pelvis[0])
            data += st.pack(float_format,self.pelvis[1])
            data += st.pack(float_format,self.pelvis[2])
        if url > 0 :
            data += st.pack(float_format,self.url[0])
            data += st.pack(float_format,self.url[1])
            data += st.pack(float_format,self.url[2])
        if lrl > 0 :
            data += st.pack(float_format,self.lrl[0])
            data += st.pack(float_format,self.lrl[2])
        if ull > 0 :
            data += st.pack(float_format,self.ull[0])
            data += st.pack(float_format,self.ull[1])
            data += st.pack(float_format,self.ull[2])
        if lll > 0 :
            data += st.pack(float_format,self.lll[0])
            data += st.pack(float_format,self.lll[2])
        if hrt > 0 :
            data += st.pack('b', self.heartrate)
        packet = preamble+tstamp+flag+data
        packet += st.pack('B', crc8(packet))
        return packet
    
    def encode_as_text(self):
        '''
            Encodes the MTBF data packet in a text representation.
            
            :Returns:
                text representation of MTBF data
        '''
        mtbf_str = ''
        mtbf_str += '{} '.format(self.timestamp) 
        if self.has_torso() :
            mtbf_str += '{0} {1} {2} '.format(self.torso[0], self.torso[1], self.torso[2]) 
        else :
            mtbf_str += '0 0 0 ' 
        if self.has_right_arm() :
            mtbf_str += '{0} {1} {2} {3} {4} '.format(self.ur[0], self.ur[1], self.ur[2], self.lr[0], self.lr[2]) 
        else :
            mtbf_str += '0 0 0 0 0 ' 
        if self.has_left_arm() :
            mtbf_str += '{0} {1} {2} {3} {4} '.format(self.ul[0], self.ul[1], self.ul[2], self.ll[0], self.ll[2]) 
        else :
            mtbf_str += '0 0 0 0 0 ' 
        if self.has_pelvis() :
            mtbf_str += '{0} {1} {2} '.format(self.pelvis[0], self.pelvis[1], self.pelvis[2]) 
        else :
            mtbf_str += '0 0 0 ' 
        if self.has_right_leg() :
            mtbf_str += '{0} {1} {2} {3} {4} '.format(self.url[0], self.url[1], self.url[2], self.lrl[0], self.lrl[2]) 
        else :
            mtbf_str += '0 0 0 0 0 ' 
        if self.has_left_leg() :
            mtbf_str += '{0} {1} {2} {3} {4} '.format(self.ull[0], self.ull[1], self.ull[2], self.lll[0], self.lll[2]) 
        else :
            mtbf_str += '0 0 0 0 0 '
        return mtbf_str[:-1]
    
    def has_torso(self):
        return testBit(self.flags, 0)
   
    def has_ur(self):
        return testBit(self.flags, 1)
       
    def has_lr(self):
        return testBit(self.flags, 2)
    
    def has_right_arm(self):
        return self.has_lr() and self.has_ur()
    
    def has_ul(self):
        return testBit(self.flags, 3)
       
    def has_ll(self):
        return testBit(self.flags, 4)
    
    def has_left_arm(self):
        return self.has_ll() and self.has_ul()
        
    def has_activity(self):
        return testBit(self.flags, 5)
    
    def has_intensity(self):
        return testBit(self.flags, 6)
    
    def has_info(self):
        return testBit(self.flags, 7)
    
    def has_warning(self):
        return testBit(self.flags, 8)

    def has_mode(self):
        return testBit(self.flags, 9)
    
    def has_sensor(self):
        return testBit(self.flags, 10)
    
    def has_pelvis(self):
        return testBit(self.flags, 11)
    
    def has_url(self):
        return testBit(self.flags, 12)
       
    def has_lrl(self):
        return testBit(self.flags, 13)

    def has_right_leg(self):
        return self.has_lrl() and self.has_url()

    def has_ull(self):
        return testBit(self.flags, 14)
       
    def has_lll(self):
        return testBit(self.flags, 15)

    def has_left_leg(self):
        return self.has_lll() and self.has_ull()
   
    def has_heartrate(self):
        return testBit(self.flags, 16)
    
    
    @staticmethod
    def create_from_tcp_packet(packet):
        return MTBFData(packet[:7], packet)
    @staticmethod
    def create_from_file_stream(stream):
        return MTBFData(stream[:7], stream)

    
    preamble = property(get_preamble, set_preamble, del_preamble, "Preamble of MTBF format.")
    warningFlag = property(get_warning_flag, set_warning_flag, del_warning_flag, "Set of predefined warnings,"
                                                                               + " one per bit")
    activityFlag = property(get_activity_flag, set_activity_flag, del_activity_flag, "Activity id (e.g. walking," + 
                                                                                     " running, cycling, ...)")
    intensityFlag = property(get_intensity_flag, set_intensity_flag, del_intensity_flag, "Intensity of the activity.")
    info = property(get_info, set_info, del_info, "Used to send simple information. Bit 0 reserved for" + 
                                                  " repetition completed.")
    sensor = property(get_sensor, set_sensor, del_sensor, "Defines the number of available sensors.")
    torso = property(get_torso, set_torso, del_torso, "Yaw, pitch, roll angles of torso in rad.")
    ur = property(get_ur, set_ur, del_ur, "Yaw, pitch, roll angles of upper right shoulder in rad.")
    lr = property(get_lr, set_lr, del_lr, "Yaw, roll angles of right elbow in rad.")
    ul = property(get_ul, set_ul, del_ul, "Yaw, pitch, roll angles of upper left shoulder in rad.")
    ll = property(get_ll, set_ll, del_ll, "Yaw, roll angles of left elbow in rad.")
    url = property(get_url, set_url, del_url, "Yaw, pitch, roll angles of right hip (upper right leg) in rad.")
    lrl = property(get_lrl, set_lrl, del_lrl, "Yaw, roll angles of right knee (lower right leg) in rad.")
    ull = property(get_ull, set_ull, del_ull, "Yaw, pitch, roll angles of left hip (upper left left) in rad.")
    lll = property(get_lll, set_lll, del_lll, "Yaw, roll angles of left knee (lower left leg) in rad.")
    pelvis = property(get_pelvis, set_pelvis, del_pelvis, "Yaw, pitch, roll angles of pelvis in rad.")
    mode = property(get_mode, set_mode, del_mode, "Number indicating what mode has been started (0xFFFF is reserved to"
                                               + " indicate failure to execute command.) This is used to keep the "
                                               + " information, which exercises were executed in order to be able "
                                               + " to reconstruct the training session during replay.")
    heartrate = property(get_heartrate, set_heartrate, del_heartrate, "Heart rate in bpm.")
    timestamp = property(get_timestamp, set_timestamp, del_timestamp, "Timestamp")

def configure_signal_3DOF(vect, data, bp, selection):
    if selection is None :
        vect.extend(data)
    else :
        if '{0}:x'.format(bp) in selection: 
            vect.append(data[0])
        if '{0}:y'.format(bp) in selection : 
            vect.append(data[1])
        if '{0}:z'.format(bp) in selection : 
            vect.append(data[2])

def configure_signal_5DOF(vect, data, bp, selection):
    if selection is None :
        vect.extend(data[0])
        vect.append(data[1][0])
        vect.append(data[1][2])
    else :
        if '{0}:x'.format(BODY_LABEL['{0}:{1}'.format(bp,'upper')]) in selection: 
            vect.append(data[0][0])
        if '{0}:y'.format(BODY_LABEL['{0}:{1}'.format(bp,'upper')])  in selection : 
            vect.append(data[0][1])
        if '{0}:z'.format(BODY_LABEL['{0}:{1}'.format(bp,'upper')])  in selection : 
            vect.append(data[0][2])
        if '{0}:x'.format(BODY_LABEL['{0}:{1}'.format(bp,'lower')])  in selection : 
            vect.append(data[1][0])
        if '{0}:z'.format(BODY_LABEL['{0}:{1}'.format(bp,'lower')])  in selection :
            vect.append(data[1][2])

def configure_signalconfig_3DOF(config, bp, pos):
    config[pos] = '{0}:{1}'.format(BODY_LABEL['{0}'.format(bp)], 'x'); pos += 1
    config[pos] = '{0}:{1}'.format(BODY_LABEL['{0}'.format(bp)], 'y'); pos += 1
    config[pos] = '{0}:{1}'.format(BODY_LABEL['{0}'.format(bp)], 'z'); pos += 1
    
def configure_signalconfig_5DOF(config, bp, pos):
    config[pos] = '{0}:{1}'.format(BODY_LABEL['{0}:{1}'.format(bp,'upper')], 'x'); pos += 1
    config[pos] = '{0}:{1}'.format(BODY_LABEL['{0}:{1}'.format(bp,'upper')], 'y'); pos += 1
    config[pos] = '{0}:{1}'.format(BODY_LABEL['{0}:{1}'.format(bp,'upper')], 'z'); pos += 1
    config[pos] = '{0}:{1}'.format(BODY_LABEL['{0}:{1}'.format(bp,'lower')], 'x'); pos += 1
    config[pos] = '{0}:{1}'.format(BODY_LABEL['{0}:{1}'.format(bp,'lower')], 'z'); pos += 1
    
class MTBFStream(object):
    '''
        MTBF stream contains the a stream of MTBF messages.
        The stream can be static or dynamic.

        The MTBF stream contains the motion data recorded by IMU body sensors.
                     
                             ooooo
                             o   o
                             ooooo
                               |
                           O--------0            ---
                     ES   /    |     \            |
                        x/    xO      \ x         |
                 elbow  O      | TORSO 0          |
                       /       |        \         |
                EW    /        |         \        |
                    x/     O---O---O      \ x     | SHOHeight
             wrist  O      |PELVIS |       O      |
                       HK  |       |              |
                          x|       |x             |
                           O       O  Knee        |
                           |       |              |
                       KF  |       |              |
                           |       |              |
                         x |       |  x           |   (x) marks the position of the sensors
                       ----O       O----         ---
       
    '''
    def __init__(self, dynamic=False, relay_mtbf=False, buffer_size=0, cached=False, MAXIMUM_IN_MEMORY=6000):
        '''
            Initializes a MTBF stream.
            
            :Parameters:
                dynamic     - flag if data is dynamic, so data can be constantly added
                relay_mtbf  - flag if the data is relayed to the MTBF
                buffer_size - size of the buffer
                cached      - flag if the motion is cached  
        
        '''
        self.current = 0
        self.mtbf_data = []
        self.app_mtbf_data = self.mtbf_data.append
        self.set_dynamic(dynamic)
        self.relay_mtbf = relay_mtbf
        self.buffer_size = buffer_size 
        self.real_length = 0
        self.index_begin_mem = -1
        self.index_end_mem   = -1
        self.MAXIMUM_IN_MEMORY = MAXIMUM_IN_MEMORY
        self.cached = cached and dynamic # only if stream is dynamic and cached
        if self.cached :
            self._init_cache_() 
        self.listener = []
        self.mtbf_listener = []

    def _init_cache_(self):
        '''
        Initialises the internal cache file.
        '''
        self.cachefile_name = tempfile.mktemp() + '.mtbf'
        self.cachefile = open(self.cachefile_name, "w+b")
        log.debug("Initialising cache file {}.".format(self.cachefile_name))
        
    def is_dynamic(self):
        return self.__dynamic

    def set_dynamic(self, value):
        self.__dynamic = value

    def del_dynamic(self):
        del self.__dynamic

    def close(self):
        '''
        Closes the stream.
        '''
        raise NotImplementedError
    
    def open_stream(self):
        '''
        Opens the stream.
        '''
        raise NotImplementedError
    
    def clear(self):
        '''
        Clears the internal data structures.
        '''
        self.mtbf_data = []
        self.app_mtbf_data = self.mtbf_data.append
        self.index_begin_mem = -1
        self.index_end_mem = -1
    
    def reload_from_cache(self):
        '''
            Reloads data if stream is in file cache mode.
        '''
        self.cached = False # now we turn off the feature
        self.cachefile.close()
        try : 
            self.clear()
            f = MTBFFileStream(self.cachefile_name)
            f.open_stream()
            self.mtbf_data = f.mtbf_data
        finally :
            log.error("Reload from cache failed.")
            
    def crop_to(self, begin, end):
        '''
            Crops the MTBF stream.
            :Parameters:
                begin - begin index for crop
                end   - end index
        '''
        assert begin > 0 and end > begin and end < len(self)
        self.mtbf_data = self.mtbf_data[begin:end]
        
    def remove(self, index):
        '''
         Removes mtbf data a specific index.
         :Paramters:
             index - index position
        '''
        if index >= 0 and index < len(self.mtbf_data) :
            del self.mtbf_data[index]
        
    def add_listener(self, listener_func):
        '''
            Adds a listener function.
            :Parameter:
                listener_func
        '''
        self.listener.append(listener_func)

    def remove_listener(self, listener_func):
        '''
            Removes a listener function.
            :Parameter:
                listener_func
        '''
        if listener_func in self.listener :
            try :
                self.listener.remove(listener_func)
            finally:
                pass # ignore: but should never happen    

    def add_mtbf_listener(self, listener_func):
        '''
            Adds a listener function.
            :Parameter:
                listener_func
        '''
        self.mtbf_listener.append(listener_func)
        
    def remove_mtbf_listener(self, listener_func):
        '''
            Removes a mtbf listener function.
            :Parameter:
                listener_func
        '''
        if listener_func in self.mtbf_listener :
            try :
                self.mtbf_listener.remove(listener_func)
            finally:
                pass # ignore: but should never happen   
            
    def __fire_event__(self, mtbf):
        for list_func in self.listener :
            list_func(mtbf)
            
    def __fire_mtbf_event__(self, mtbf):
        for list_func in self.mtbf_listener :
            list_func(mtbf)
            
    def update(self, mtbf, append=True):
        '''
            Receives an mtbf package.
                mtbf - package
                appendtype - 0 is right append
                             1 is left append
                             2 is no append
        '''
        if append : self.app_mtbf_data(mtbf)
        if self.relay_mtbf :
            self.__fire_event__(self[-1])
            self.__fire_mtbf_event__(mtbf)
        if self.buffer_size > 0 and self.buffer_size < len(self.mtbf_data) :
            self.remove(0)
        if self.cached :
            self.cachefile.write(mtbf.encode_binary())
            
    def save_stream(self, fname):
        '''
            Saves the stream in a file.
            :Parameters:
                fname : filename
        '''
        try : 
            f = open(fname, 'wb')
            for mtbf_package in self.mtbf_data :
                f.write(mtbf_package.encode_binary())
        finally :
            if f != None : f.close()
            
    def save_stream_as_text(self, fname):
        '''
            Saves stream as text file.
            :Parameters:
                fname - name of file
        '''
        try : 
            f = open(fname, 'wb')
            for mtbf_package in self.mtbf_data :
                f.write(mtbf_package.encode_as_text() + '\n')
        finally :
            if f != None : f.close()
    
    def save_stream_as_mat(self, matfile):
        '''
            Saves stream as Matlab file.
            :Parameters:
                fname - name of file
        '''
        mat_dict = {}#MotionData.clear(self)
        
        if self.has_left_arm() :
            mat_dict['Left_Arm'] =  np.array([[t[0][0], t[0][1], t[0][2], t[1][0], t[1][2] ]for t in self.get_left_arm()]) 
        if self.has_right_arm() :
            mat_dict['Right_Arm'] = np.array([[t[0][0], t[0][1], t[0][2], t[1][0], t[1][2] ]for t in self.get_right_arm()])
        if self.has_torso() :
            mat_dict['Torso'] = np.array([[t[0], t[1], t[2]] for t in self.get_torso()])
        if self.has_left_leg() :
            mat_dict['Left_Arm'] =  np.array([[t[0][0], t[0][1], t[0][2], t[1][0], t[1][2] ]for t in self.get_left_leg()]) 
        if self.has_right_leg() :
            mat_dict['Right_Arm'] = np.array([[t[0][0], t[0][1], t[0][2], t[1][0], t[1][2] ]for t in self.get_right_leg()])
        if self.has_pelvis() :
            mat_dict['Pelvis'] = np.array([[t[0], t[1], t[2]] for t in self.get_pelvis()])
        matio.savemat(matfile, mat_dict, oned_as='row')
        
    def write(self, filename):
        self.save_stream(filename)
        
    def read(self, begin, end):
        pass
    
    def __getitem__(self, key) :
        if type(key) == slice :
            stream = MTBFStream(self.dynamic)
            stream.mtbf_data = self.mtbf_data[key]
            return stream
        elif type(key) == list :
            return self.get_signal_array(key)
        elif type(key) == tuple and len(key) == 2 :
            return self[key[0]:key[1]]
        elif type(key) == int :
            return self.mtbf_data[key]
    def __len__(self):
        return len(self.mtbf_data)

    def get_timestamps(self):
        ''' 
            Returns the synchronized timestamps for each body part.
            :Returns:
                list of float timestamps
        '''
        return [m.get_timestamp() for m in self.mtbf_data] 
    
    def get_normalized_timestamps(self):
        ''' 
            Returns the normalized synchronized timestamps for each body part.
            :Returns:
                list of float timestamps : starting with 0.0
        '''
        return [ti - self.mtbf_data[0].get_timestamp() for ti in self.get_timestamps()]
    
    def get_signal_array(self, selection=None):
        '''
            Returns the motion signal as numpy array.ap
            :Parameters:
                selection - list with string 
            :Returns:
                
        '''
        signal = []
        sig_append = signal.append
        torso = []; pelvis = []
        right_arm = []; left_arm = []
        right_leg = []; left_leg = [] 
        if self.has_torso() : torso = self.get_torso()
        if self.has_pelvis() : pelvis = self.get_pelvis()
        if self.has_right_arm() : right_arm = self.get_right_arm()
        if self.has_left_arm() : left_arm = self.get_left_arm()
        if self.has_right_leg() : right_leg = self.get_right_leg()
        if self.has_left_leg() : left_leg = self.get_left_leg()
            
        for i in xrange(len(self.mtbf_data)) :
            vect = []
            if self.has_torso() :
                configure_signal_3DOF(vect, torso[i], 'torso', selection)
            if self.has_pelvis() :
                configure_signal_3DOF(vect, pelvis[i], 'pelvis', selection)
            if self.has_right_arm() :
                configure_signal_5DOF(vect, right_arm[i], 'rightarm', selection) 
            if self.has_left_arm() :
                configure_signal_5DOF(vect, left_arm[i], 'leftarm', selection)  
            if self.has_right_leg() :
                configure_signal_5DOF(vect, right_leg[i], 'rightleg', selection) 
            if self.has_left_leg() :
                configure_signal_5DOF(vect, left_leg[i], 'leftleg', selection)
            sig_append(vect)
        return np.array(signal)
    
    def get_channel_mapping(self):
        '''
            Corresponding channel mapping for signal array.
            :Returns:
                dict : {channel-id : limb channel}
        '''
        signal_mapping = {}
        pos = 0
        if self.has_torso() :     configure_signalconfig_3DOF(signal_mapping, 'torso', pos);    pos += 3
        if self.has_pelvis() :    configure_signalconfig_3DOF(signal_mapping, 'pelvis', pos);   pos += 3
        if self.has_right_arm() : configure_signalconfig_5DOF(signal_mapping, 'rightarm', pos); pos += 5
        if self.has_left_arm() :  configure_signalconfig_5DOF(signal_mapping, 'leftarm', pos);  pos += 5  
        if self.has_right_leg() : configure_signalconfig_5DOF(signal_mapping, 'rightleg', pos); pos += 5 
        if self.has_left_leg() :  configure_signalconfig_5DOF(signal_mapping, 'leftleg', pos);  pos += 5
        return signal_mapping 

    def number_of_signal_channels(self):
        '''
            Returns the number of channels, which are stored in the motion data.
            :Returns:
                number of signals
        '''
        channels = 0
        if len(self.mtbf_data) == 0 :
            return channels
        if self.mtbf_data[0].has_torso() : channels +=3
        if self.mtbf_data[0].has_pelvis() : channels +=3
        if self.mtbf_data[0].has_right_arm() : channels +=5
        if self.mtbf_data[0].has_left_arm() : channels +=5
        if self.mtbf_data[0].has_right_leg() : channels +=5 
        if self.mtbf_data[0].has_left_leg() : channels +=5
        return channels
        
    def get_torso(self):
        ''' 
            Returns euler angles for torso.
            :Returns:
                list of Eulerangles
        '''
        return [m.get_torso() for m in self.mtbf_data] 
    def get_pelvis(self):
        ''' 
            Returns euler angles for pelvis.
            :Returns:
                list of Eulerangles
        '''
        return [m.get_pelvis() for m in self.mtbf_data] 
    
    def get_right_arm(self):
        ''' 
            Returns euler angles for right arm.
            :Returns:
                list of tuples (3DOF upper arm euler angle (xyz), 2DOF lower arm (xz))] 
        '''
        return [(m.get_ur(), m.get_lr()) for m in self.mtbf_data] 
    
    def get_left_arm(self):
        ''' 
            Returns euler angles for left arm.
            :Returns:
                list of tuples (3DOF upper arm euler angle (xyz), 2DOF lower arm (xz))] 
        '''
        return [(m.get_ul(), m.get_ll()) for m in self.mtbf_data] 
    
    def get_right_leg(self):
        ''' 
            Returns euler angles for right leg.self
            :Returns:ap
                list of tuples (3DOF upper arm euler angle (xyz), 2DOF lower arm (xz))] 
        '''
        return [(m.get_url(), m.get_lrl()) for m in self.mtbf_data] 
    
    def get_left_leg(self):
        ''' 
            Returns euler angles for left leg.
            :Returns:
                list of tuples (3DOF upper arm euler angle (xyz), 2DOF lower arm (xz))] 
        '''
        return [(m.get_ull(), m.get_lll()) for m in self.mtbf_data] 
  
    def has_right_arm(self):
        '''
            Checks if the MTBF stream contains data from right arm.
            :Returns:
                flag if data is available.
        '''
        if len(self.mtbf_data) == 0 :
            return False
        return self.mtbf_data[0].has_right_arm()
    
    def has_left_arm(self):
        '''
            Checks if the MTBF stream contains data from left arm.
            :Returns:
                flag if data is available.
        '''
        if len(self.mtbf_data) == 0 :
            return False
        return self.mtbf_data[0].has_left_arm()
    
    def has_torso(self):
        '''
            Checks if the MTBF stream contains data from torso.self
            :Returns:
                flag if data is available.
        '''
        if len(self.mtbf_data) == 0 :
            return False
        return self.mtbf_data[0].has_torso()
    
    def has_pelvis(self):
        '''
            Checks if the MTBF stream contains data from pelvis.
            :Returns:
                flag if data is available.
        '''
        if len(self.mtbf_data) == 0 :
            return False
        return self.mtbf_data[0].has_pelvis()

    def has_right_leg(self):
        '''
            Checks if the MTBF stream contains data from right leg.
            :Returns:
                flag if data is available.
        '''
        if len(self.mtbf_data) == 0 :
            return False
        return self.mtbf_data[0].has_right_leg()
    def has_left_leg(self):
        '''
            Checks if the MTBF stream contains data from left leg.
            :Returns:
                flag if data is available.
        '''
        if len(self.mtbf_data) == 0 :
            return False
        return self.mtbf_data[0].has_left_leg()
    
    
    ''' ----------- Iterator ---------- '''
    def __iter__(self):
        return self

    def next(self):
        try :
            if len(self.mtbf_data) == 0 : raise StopIteration
            data = self[self.current]
            self.current += 1
        except :
            self.current = 0
            raise StopIteration
        return data
    
    
    dynamic = property(is_dynamic, set_dynamic, del_dynamic, "Indicates if the stream is static (false) or dynamic (true)")

        
class MTBFFileStream(MTBFStream): 
    '''
        A static MTBF File stream parser.
    '''
    def __init__(self, f, relay_mtbf=False,  MAXIMUM_IN_MEMORY=30000):
        MTBFStream.__init__(self, False, relay_mtbf, MAXIMUM_IN_MEMORY=MAXIMUM_IN_MEMORY)
        self.file = f
        
    def open_stream(self, begin=0, end=-1):
        log.debug('Load MTBF file : {0}.'.format(self.file))
        self.read(begin, end)
        
    def read(self, begin=0, end=-1, readbuffer=1024):
        '''
            Opens the file stream and parses the data.
            :Parameters:
                begin - index, where it the parser starts to read
                end   - index, where it the parser ends with reading
        '''
        global MTBF_BYTE_MINSIZE, log
        # Sometimes typesafety would be nice
        if begin == 0 and end == -1 : end = self.MAXIMUM_IN_MEMORY
        begin = int(begin)
        end = int(end)
        offset = max([begin, self.index_begin_mem])
        if not os.path.isfile(self.file) :
            log.error('MTBF file does not exist. file : {0}'.format(self.file))
            raise Exception('MTBF file does not exist. file : {0}'.format(self.file))
        if begin < 0 or (not end == -1 and begin >= end) :
            raise Exception('Unvalid boundaries [begin := {} end :={}].'.format(begin, end))
       
        samples = self.read_number_of_samples()
        if end - begin > self.MAXIMUM_IN_MEMORY or samples > self.MAXIMUM_IN_MEMORY and end == -1 :
            log.warn('Reducing end, as it exceeds the MAXIMUM_SAMPLES_IN_MEMORY ({}) policy. end {} -> new end'.format(self.MAXIMUM_IN_MEMORY,
                                                                                                                       end,
                                                                                                                       begin + self.MAXIMUM_IN_MEMORY))
            end = begin + self.MAXIMUM_IN_MEMORY
        in_memory_strategy = self.index_begin_mem != -1 and self.index_end_mem != -1 \
                        and not end < self.index_begin_mem and not begin > self.index_end_mem
        
        if not in_memory_strategy :
            self.clear()
        elif in_memory_strategy and begin > self.index_begin_mem : # Cut useless begin part 
            del self.mtbf_data[:begin - self.index_begin_mem]

        with open(self.file, 'rb') as f :
            # ALL MTBF packages
            done = False
            buf = ''
            before_mem = []
            pcount = 0
            while not done : # abort when we are at the end of file or out of boundary
                buf += f.read(readbuffer)
                done = len(buf) < readbuffer
                while len(buf) >= (MTBF_BYTE_MINSIZE) :
                    packet_end = buf.find('MTBF', 7)
                    packet = buf[:packet_end]
                    if packet_end == -1 and done : # last package
                        self.update(MTBFData.create_from_file_stream(buf))
                    elif not packet_end == -1 and pcount >= begin :
                        if in_memory_strategy and pcount < self.index_begin_mem :
                            mtbf_data = MTBFData.create_from_file_stream(packet)
                            before_mem.append(mtbf_data)
                            self.update(mtbf_data, False)
                        elif in_memory_strategy \
                        and pcount >= self.index_begin_mem and pcount < self.index_end_mem :
                            self.update(self.mtbf_data[pcount - offset], False)
                        else :
                            self.update(MTBFData.create_from_file_stream(packet))
                    elif packet_end == -1 :
                        break
                    pcount += 1
                    if pcount == end : # abortion criterion
                        done = True
                        break
                    buf = buf[packet_end:]
            self.update_mem(begin, pcount, before_mem)
            
    def update_mem(self, begin, end, before_mem) :
        '''
           Updates the internal cache.
           :Parameters:
               begin - begin of cache
               end   - end of cache
        '''
        self.index_begin_mem = begin
        self.index_end_mem = end
        if len(before_mem) > 0 : 
            # adding the collected data before the in memory part 
            before_mem.extend(self.mtbf_data)
            self.mtbf_data = before_mem
        # Cut parts away if there was too much in memory
        self.mtbf_data = self.mtbf_data[:(end - begin) + 1]
        self.app_mtbf_data = self.mtbf_data.append
            
    def get_memory_begin_index(self):
        return self.index_begin_mem
        

    def read_number_of_samples(self):
        '''
            Reads the number of MTBF samples from the file.
            :Returns:
                number of samples in file.
        '''
        if not os.path.isfile(self.file) :
            log.error('MTBF file does not exist. file : {0}'.format(self.file))
            raise Exception('MTBF file does not exist. file : {0}'.format(self.file))
        with open(self.file, 'rb') as f :
            buf = f.read()
            self.real_length = buf.count('MTBF')
        return self.real_length
            
    def start_relay_mtbf(self):
        '''
            Starts the relay functionality. 
            Registered listeners will be informed with 100 Hz.
        '''
        for mtbf in self.mtbf_data :
            self.__fire_event__(mtbf)
            time.sleep(0.01)
        
    def get_data(self):
        return self.data
    
