"""Read verified MSO44 TIME,CH1,CH2,CH3 CSV exports without resampling.

CSV channels are already in V; do not apply WFM integer scaling again.
SHA256 records file identity; CSV has no instrument checksum like WFM.
"""
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CsvInfo:
    path: str
    samples: int
    dt_s: float
    t0_s: float
    y_unit: str = 'V'

    @property
    def sample_rate_hz(self):
        return 1/self.dt_s


class CsvChannel:
    def __init__(self,record,column):
        self.record=record
        self.column=column
        self.info=record.info

    def values(self,start=0,stop=None,step=1):
        return self.record.data[start:stop:step,self.column]

    def times(self,start=0,stop=None,step=1):
        return self.record.data[start:stop:step,0]


class CsvRecord:
    def __init__(self,path):
        path=Path(path)
        with path.open(encoding='utf-8-sig',newline='') as stream:
            metadata=[]
            for line,row in enumerate(csv.reader(stream),1):
                if row and row[0].strip().upper()=='TIME':
                    columns=[x.strip().upper() for x in row]; break
                metadata.append(row)
            else: raise ValueError('CSV TIME header missing')
        if columns!=['TIME','CH1','CH2','CH3']:
            raise ValueError('CSV requires TIME,CH1,CH2,CH3; found '+str(columns))
        def fields(name):
            return [row[i+1].strip() for row in metadata for i,x in enumerate(row[:-1]) if x.strip()==name]
        if fields('Model')!=['MSO44']:
            raise ValueError('Only verified MSO44 CSV layout supported')
        if fields('Vertical Units')!=['V']*3 or fields('Horizontal Units')!=['s']*3:
            raise ValueError('Expected volts and seconds for all CSV channels')
        intervals=[float(x) for x in fields('Sample Interval')]
        lengths=[int(x) for x in fields('Record Length')]
        if len(intervals)!=3 or not all(np.isfinite(x) for x in intervals) or any(x!=intervals[0] for x in intervals) or intervals[0]<=0:
            raise ValueError('CSV channels have inconsistent sample intervals')
        data=np.loadtxt(path,delimiter=',',skiprows=line,encoding='utf-8-sig')
        if data.ndim!=2 or data.shape[1]!=4 or not np.isfinite(data).all():
            raise ValueError('Invalid CSV numerical data')
        if lengths!=[len(data)]*3:
            raise ValueError('CSV length differs from instrument header')
        dt=intervals[0]
        tolerance=max(1e-12,2e-8*float(np.max(np.abs(data[:,0]))))
        if (np.any(np.diff(data[:,0])<=0) or
                np.max(np.abs(data[:,0]-(data[0,0]+np.arange(len(data))*dt)))>tolerance):
            raise ValueError('CSV time grid disagrees with instrument sample interval')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
        self.sha256=digest.hexdigest()
        self.data=data
        self.info=CsvInfo(str(path),len(data),dt,float(data[0,0]))

    @property
    def waves(self):
        # Do not store channels on the parent: that would retain large arrays
        # through a reference cycle while reading many files in succession.
        return {role:CsvChannel(self,column) for role,column in [('uhf',1),('phase_ref',2),('hfct',3)]}

    def check_phase(self):
        """Same low-pass and positive crossings as PRPD, without plotting imports."""
        from scipy.signal import butter,sosfiltfilt
        ref=self.data[:,2]
        smooth=sosfiltfilt(butter(2,200/(self.info.sample_rate_hz/2),btype='low',output='sos'),ref-np.median(ref))
        indices=np.flatnonzero((smooth[:-1]<0)&(smooth[1:]>=0))+1
        if len(indices)<2:
            raise ValueError('CSV has fewer than two positive phase-reference crossings')
        frac=-smooth[indices-1]/(smooth[indices]-smooth[indices-1])
        times=self.data[indices-1,0]+frac*(self.data[indices,0]-self.data[indices-1,0])
        if not np.any((np.diff(times)>.015)&(np.diff(times)<.025)):
            raise ValueError('CSV has no phase-reference period in 15-25 ms')
        return len(indices)
