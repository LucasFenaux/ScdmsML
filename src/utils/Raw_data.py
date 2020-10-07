from __future__ import division
import warnings
import pandas as pd
import gzip
import scipy.io
import scipy.stats
import os, sys
import numpy

class raw(object):
    def __init__(self, filename):
        self.filename = filename
    
    def __enter__(self):
        #self.gzipfile = os.popen('gzip -dc %s'%self.filename)
        self.gzipfile = gzip.open(self.filename)
        self.read_file_header()
        
        self.header = numpy.zeros(2, dtype=self.endian + 'u4')
        self.trace_header = numpy.zeros(12, dtype=self.endian + 'u4')
        self.admin_record = numpy.zeros(6, dtype=self.endian + 'u4')
        self.phonon_channel_config_record = numpy.zeros(11, dtype=self.endian+'i4')
        self.charge_channel_config_record = numpy.zeros(8, dtype=self.endian+'i4')
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gzipfile.close()
        
    def read_file_header(self):
        header1 = numpy.ndarray(8, dtype='u1')
        self.gzipfile.readinto(header1.data)
        if all(header1[0:4] == numpy.array([1,2,3,4])):
            self.endian = '>'
        elif all(header1[0:4] == numpy.array([4,3,2,1])):
            self.endian = '<'
        else:
            warnings.warn("This doesn't look like a Soudan data file -- no discernible "
                          "endianness indicator.  Assuming little-endian and trying to "
                          "forge ahead.  Resulting data might be nonsense.")
            self.endian = '<'
        self.file_header = numpy.ndarray(2, buffer=header1.data, dtype=self.endian + 'u4')

    def read_header(self):
        self.n_read = self.gzipfile.readinto(self.header.data)
        if self.n_read == 0:
            raise EOFError
        if self.header[0] == 0x00010000:
            self.header_type = 0
            self.parse_detector_config()
        elif ((self.header[0] & 0xffff0000)>>16) == 0xa980:
            self.header_type = 1
            self.parse_event_header()
        elif self.header[0] in {0x00000002, 0x00000011, 0x00000021, 0x00000060, 0x00000080, 0x00000081}:
            self.header_type = 2
            self.parse_logical_record_header()
        else:
            raise ValueError("unexpected header code", hex(self.header[0]))
        
    def parse_detector_config(self):
        # Just skip past this for now.
        self.detector_config_length = self.header[1]
        #self.gzipfile.read(self.detector_config_length)
        self.n_read = 2
        while self.n_read <= self.detector_config_length:
            self.n_read += self.gzipfile.readinto(self.header.data)
            if self.n_read == 0:
                raise EOFError
            elif self.header[0] == 0x00010001:
                self.parse_phonon_channel_config_record()
            elif self.header[0] == 0x00010002:
                self.parse_charge_channel_config_record()
            else:
                raise ValueError("unexpected detector config header code", hex(self.header[0]))
                
    def parse_phonon_channel_config_record(self):
        assert self.header[1] == 11*4
        self.n_read += self.gzipfile.readinto(self.phonon_channel_config_record.data)
        #print(self.phonon_channel_config_record)
            
    def parse_charge_channel_config_record(self):
        assert self.header[1] == 8*4
        self.n_read += self.gzipfile.readinto(self.charge_channel_config_record.data)
        #print(self.charge_channel_config_record)
        
    def parse_event_header(self):
        self.event_type = (self.header[0] & 0x000000ff)
        self.event_category = (self.header[0] & 0x00000f00) >> 8
        self.event_class = (self.header[0] & 0x0000f000) >> 12
        self.event_size = self.header[1]
        
    def parse_logical_record_header(self):
        self.record_type, self.record_length = self.header
        
    def read_admin_record(self):
        self.gzipfile.readinto(self.admin_record.data)
        self.series_number_date = self.admin_record[0]
        self.series_number_time = self.admin_record[1]
        self.event_number = self.admin_record[2]
        self.event_time = self.admin_record[3]
        self.time_between = self.admin_record[4]
        self.live_time = self.admin_record[5]
        
    def read_trace_header(self):
        assert self.record_type == 0x00000011
        self.gzipfile.readinto(self.trace_header.data)
        
        assert self.trace_header[0] == 0x00000011
        assert self.trace_header[1] == 12
        self.digitizer_base_address = self.trace_header[2]
        self.digitizer_channel = self.trace_header[3]
        self.detector_code = self.trace_header[4]
        self.detector_type = self.detector_code // 1000000
        self.detector_number = (self.detector_code % 1000000) // 1000
        self.channel_number = self.detector_code % 1000
        assert self.trace_header[5] == 0x00000012
        assert self.trace_header[6] == 12
        self.t0 = self.trace_header[7]
        self.delta_t = self.trace_header[8]
        self.num_points = self.trace_header[9]
        assert self.trace_header[10] == 0x00000013
        self.num_samples = self.trace_header[11]
        
    def read_trace(self, buf=None):
        #if buf is not None:
            #assert len(buf) == self.num_samples, (len(buf), self.num_samples)
        #else:
        if buf is None:
            self.trace_data = numpy.ndarray(self.num_samples, dtype='<u2')
            buf = self.trace_data.data
        self.gzipfile.readinto(buf)
        

def read_file(fname,detlist=(4,),chanlist=(2,3,4,5,8,9,10,11),n_samples=4096,n_to_read=100000,quiet=True):
    n_ev = 0
    n_read = 0
    #n_to_read = 100000 #large number to read full dump
    
    traces = numpy.zeros((n_to_read, n_samples), dtype='<u2')
    detnum = numpy.zeros((n_to_read), dtype='u1')
    channum = numpy.zeros((n_to_read), dtype='u1')
    eventnum = numpy.zeros((n_to_read), dtype='u4')
    eventtype = numpy.zeros((n_to_read), dtype='u1')
    eventcat = numpy.zeros((n_to_read), dtype='u1')

    if not quiet: print(fname)
    sys.stdout.flush()
    with raw(fname) as reader:
        while True:
            #print n_read, 
            try:
                reader.read_header()
                #print reader.header_type,
            except EOFError:
                break
            except ValueError:
                continue
            if n_read >= traces.shape[0]:
                traces.resize((n_read*2, traces.shape[1]))
                detnum.resize((n_read*2))
                channum.resize((n_read*2))
                eventnum.resize((n_read*2))
                eventtype.resize((n_read*2))
                eventcat.resize((n_read*2))
            if reader.header_type == 2:
                #print hex(reader.record_type),
                if reader.record_type == 0x02:
                    #print 'admin record',
                    reader.read_admin_record()
                    #print reader.event_number, reader.event_type, reader.event_category, reader.event_class,
                elif reader.record_type == 0x11:
                    reader.read_trace_header()
                    #print reader.detector_type, reader.detector_number, reader.channel_number, reader.num_samples, reader.event_number,
                    #if reader.detector_type == 11 and 
                    if reader.detector_number in detlist and reader.channel_number in chanlist:
                        detnum[n_read] = reader.detector_number
                        channum[n_read] = reader.channel_number
                        eventnum[n_read] = reader.event_number
                        eventtype[n_read] = reader.event_type
                        eventcat[n_read] = reader.event_category
                        reader.read_trace(traces[n_read].data)
                        n_read += 1
                    else:
                        reader.gzipfile.read(reader.num_samples*2)
                else:
                    reader.gzipfile.read(reader.record_length)
            elif reader.header_type == 1:
                #assert n_ev == len(unique(eventnum[eventnum.nonzero()]))
                n_ev += 1
                #print "event header", n_ev, len(unique(eventnum)), eventnum[n_read-1],
            #print
            
    traces.resize(n_read, traces.shape[1])
    detnum.resize(n_read)
    channum.resize(n_read)
    eventnum.resize(n_read)
    eventtype.resize(n_read)
    eventcat.resize(n_read)
    #assert n_ev == len(unique(eventnum))
    df = pd.concat([pd.DataFrame({'event number':eventnum, 'detector number':detnum, 
                                  'event type': eventtype, 'channel number':channum,
                                  'event category': eventcat }),
                    pd.DataFrame(traces)],
                   axis=1).sort_values(['event number', 'detector number', 'event type', 'channel number'])
    return df


### Usage Example:
# sample_rate = 625e3 # 625 kHz
# N_samples = 4096
# sample_times = (arange(N_samples) - 512.5) / sample_rate
#
# df = read_file('/home/georgemm01/gdrive/Data/DevV1-4/raw/raw_data/01130603_0831/01130603_0831_F0001.gz')
# df = df.set_index(['event number'])#, 'channel number'])#,'detector number'])")
#
# display(df.index.values[0:3])
# print df.shape
# print numpy.shape(df.values.reshape(int(df.shape[0]/8),8,4098))
# traces=df.values.reshape(int(df.shape[0]/8),8,4098)[:,:,2:]
# display(traces[0])
