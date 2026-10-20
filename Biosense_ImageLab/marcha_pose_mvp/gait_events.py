"""Debounced gait phases and footfall registration independent of calibration."""
from dataclasses import dataclass
import numpy as np
from landmark_quality import LandmarkValidator

@dataclass(frozen=True)
class GaitOptions:
    confirmation_seconds: float = 0.06
    maximum_gap_seconds: float = 0.10
    same_side_interval_seconds: float = 0.30
    alternating_interval_seconds: float = 0.10
    measurement_wait_seconds: float = 0.20

class GaitPhaseAnalyzer:
    def __init__(self, visibility_threshold=0.45, options=None):
        self.validator = LandmarkValidator(visibility_threshold)
        self.options = options or GaitOptions()

    def analyze(self, data):
        result = {}
        for side, heel, toe in [('left',29,31),('right',30,32)]:
            centers, lowest = [], []
            for points in data.landmarks:
                visible = [index for index in (heel,toe) if self.validator.valid(points,(index,))]
                centers.append(np.mean(points[visible,:2],axis=0) if visible else None)
                lowest.append(max(points[index,1] for index in visible) if visible else np.nan)
            size = len(centers)
            valid = np.array([point is not None for point in centers], dtype=bool)
            ground_values = np.array(lowest)[valid]
            ground = float(np.percentile(ground_values,90)) if len(ground_values) else data.height
            speed = np.full(size,np.nan)
            for index in range(1,size):
                if valid[index] and valid[index-1]:
                    speed[index] = np.linalg.norm(centers[index]-centers[index-1])*data.fps
            moving = speed[np.isfinite(speed)&(speed>0)]
            speed_limit = max(150.0,float(np.percentile(moving,55))) if len(moving) else 150.0
            raw = [None if not valid[i] else bool(lowest[i]>ground-data.height*0.045 and
                       (not np.isfinite(speed[i]) or speed[i]<=speed_limit)) for i in range(size)]
            stance = np.zeros(size,dtype=bool)
            known = np.zeros(size,dtype=bool)
            events = ['']*size
            state, candidate, count, missing = None,None,0,0
            confirmation = max(2,int(np.ceil(data.fps*self.options.confirmation_seconds)))
            maximum_gap = max(1,int(round(data.fps*self.options.maximum_gap_seconds)))
            for index, value in enumerate(raw):
                if value is None:
                    missing += 1
                    candidate,count = None,0
                    if missing>maximum_gap:
                        state = None
                    stance[index] = bool(state)
                    continue
                missing = 0
                if candidate == value:
                    count += 1
                else:
                    candidate,count = value,1
                if count>=confirmation and state != value:
                    if state is not None:
                        events[index] = 'CONTATO INICIAL' if value else 'RETIRADA DO PE'
                    state = value
                stance[index] = bool(state)
                known[index] = state is not None
            result[side] = dict(stance=stance,event=events,centers=centers,valid=valid&known,
                                observed=valid,ground_px=ground,speed_limit_px_s=speed_limit)
        return result

class FootfallRegistry:
    def __init__(self, data, visibility_threshold=0.45, options=None):
        self.data = data
        self.validator = LandmarkValidator(visibility_threshold)
        self.options = options or GaitOptions()
        self.footfalls = []
        self.strides = []
        self.decisions = []
        self.last_contact = {}
        self.last_heel = {}
        self.support_confirmed = {'left':False,'right':False}

    def update(self,index,points,phases):
        updates = {}
        for side, heel, toe, ankle, other in [('left',29,31,27,28),('right',30,32,28,27)]:
            event = phases[side]['event'][index]
            if event=='RETIRADA DO PE' or not phases[side]['stance'][index]:
                self.support_confirmed[side]=False
            if event != 'CONTATO INICIAL' or not phases[side]['valid'][index]:
                continue
            previous = self.last_contact.get(side)
            if previous is not None and index-previous<self.data.fps*self.options.same_side_interval_seconds:
                self.decisions.append(dict(frame=index,side=side,accepted=False,reason='same_side_debounce'))
                continue
            last = self.footfalls[-1] if self.footfalls else None
            alternating = last is not None and last['side']!=side
            if last is not None and not alternating:
                self.decisions.append(dict(frame=index,side=side,accepted=False,reason='nonalternating_contact'))
                self.support_confirmed[side]=False
                continue
            if alternating and index-last['frame']<self.data.fps*self.options.alternating_interval_seconds:
                self.decisions.append(dict(frame=index,side=side,accepted=False,reason='alternating_debounce'))
                continue
            entry = dict(number=len(self.footfalls)+1,side=side,frame=index,position=None,
                         step_cm=None,stride_cm=None,measurement_frame=None,heel_x=None,
                         sequence='first' if last is None else 'alternating' if alternating else 'contralateral_unobserved')
            self.footfalls.append(entry)
            self.last_contact[side]=index
            self.support_confirmed[side]=True
            self.decisions.append(dict(frame=index,side=side,accepted=True,number=entry['number'],
                                       reason='confirmed_support_transition',sequence=entry['sequence']))
            updates.update(step_number=entry['number'],step_side=side)
        for entry in self.footfalls:
            if index-entry['frame']>self.data.fps*self.options.measurement_wait_seconds:
                continue
            side=entry['side']
            if not phases[side]['valid'][index] or not phases[side]['stance'][index]:
                continue
            heel,toe,ankle,other=(29,31,27,28) if side=='left' else (30,32,28,27)
            if entry['position'] is None:
                index_point=heel if self.validator.valid(points,(heel,)) else toe
                if self.validator.valid(points,(index_point,)):
                    entry['position']=points[index_point,:2].copy()
                    entry['position_source']='heel' if index_point==heel else 'toe'
            if entry['step_cm'] is None and self.data.px_per_cm:
                length=self.validator.distance(points,(ankle,other),self.data.px_per_cm,horizontal=True)
                if length is not None:
                    entry['step_cm']=length
                    entry['measurement_frame']=index
            if entry['heel_x'] is None and self.validator.valid(points,(heel,)):
                x=float(points[heel,0])
                old=self.last_heel.get(side)
                entry['heel_x']=x
                self.last_heel[side]=(entry['number'],x)
                if old is not None and self.data.px_per_cm:
                    length=abs(x-old[1])/self.data.px_per_cm
                    entry['stride_cm']=length
                    stride=dict(number=len(self.strides)+1,side=side,length_cm=length,frame=index)
                    self.strides.append(stride)
                    updates.update(stride_number=stride['number'],stride_side=side)
        if self.footfalls and self.footfalls[-1]['step_cm'] is not None:
            updates['step_length_cm']=self.footfalls[-1]['step_cm']
        if self.strides:
            updates['stride_length_cm']=self.strides[-1]['length_cm']
        return updates
