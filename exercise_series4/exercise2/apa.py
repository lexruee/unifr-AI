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

Sample call (apa and apa-include have to be extracted to the same folder in which also this script lies:
data, data_labels = read_data_config ("apa/apa00/app0032.dat")
print data
print data_labels
you can do anything with data and labels now.

'''
from __future__ import with_statement
import re
import numpy as np
   
def read_data_files(file_list):
    """
        Reads a file list of config files. 
        :Parameters:
            file_list - list of files
    
        :Returns:
            data - (x, y) sequence
            labels - labels
    """ 
    data = []
    labels = []
    # Iterate over all files
    for fi in file_list:
        d, l = read_data_config(fi)
        # Add data an according labels to the list
        data.extend(d)
        labels.extend(l)
    return data, labels
                
def read_data_config(config):
    """
        Reads a APA config file.
        :Parameter:
            config - location of config file
    """   
    # Datafile with xy coordinates
    datafile = ''
    # collect all the labels
    labels = []
    data = []
    data_labels = []
    with open(config, 'r') as f:
        for line in f :
            line = line.strip()
            # Looking for data file
            if line.startswith(".INCLUDE") :
                values = re.split(" ", line)
                # Just looking for data files
                if values[1].endswith(".dat") :
                    # .INCLUDE filename
                    datafile = values[1]
                # Looking for different segment descriptions 
            elif line.startswith('.SEGMENT') :
                # .SEGMENT CHARACTER 0 ? "0"
                values = re.split(" ", line.strip())
                if len(values) == 5 : # length should be 5
                    # .SEGMENT CHARACTER 15-16 ? "4" 
                    labels.append((int(values[4][1:-1]), values[2]))
    # Finally read data as well
    xydata = read_data(datafile)
    
    for l in labels :
        data_labels.append(l[0])
        numbers = re.split("-", l[1])
        if len(numbers) == 1 : # Just for 1 segment
            data.append(xydata[int(l[1])])
        elif len(numbers) == 2 : # multiple segments
            l = []
            for xy in xydata[int(numbers[0]):int(numbers[1])+1] :
                l.extend(xy)
            data.append(l)
    return data, data_labels
    
 
def read_data(apa_file):
    """
         Reads the data in APA format.
        :Parameters:
            apa_file - location of file
        :Returns:
            xydata - sequence of strokes with samples as numpy array (vector) for coordinates 
    """    
    xydata = []
    data = []
    with open(apa_file, 'r') as f:
        for line in f :
            line = line.strip()
            # on every pen up event all the data is already collected 
            if line == ".PEN_UP" :
                xydata.append(data)
                data = []
            else :
                values = re.split("\t", line.strip())
                if len(values) == 2 :
                    data.append(np.array([int(values[0]), int(values[1])]))               
    return xydata

