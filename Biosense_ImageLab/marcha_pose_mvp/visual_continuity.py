"""Visual continuity is isolated from measurements and never modifies input landmarks."""
from dataclasses import dataclass
import numpy as np
from joint_angles import JointAngleCalculator, LANDMARK_IDS
from landmark_quality import LandmarkValidator

@dataclass
class DisplayFrame:
    points: np.ndarray
    metrics: dict
    status: dict
    landmark_status: list
    sides: tuple

class VisualContinuity:
    def __init__(self, options):
        self.options=options
        self.sides=tuple({'Esquerdo':('left',),'Direito':('right',),'Ambos':('left','right')}.get(options.side,('left','right')))
        self.records=[]

    def build(self,sequence,fps):
        frames=[]
        validator=LandmarkValidator(self.options.visibility_threshold)
        calculator=JointAngleCalculator(self.options.visibility_threshold)
        cached=np.full((33,3),np.nan,dtype=np.float32)
        last_point=np.full(33,-np.inf)
        cached_metrics={}
        last_metric={}
        maximum=max(1,int(round(fps*self.options.visual_hold_seconds)))
        self.records=[]
        for index,points in enumerate(sequence):
            render=cached.copy()
            point_status=[]
            for landmark in range(33):
                if validator.valid(points,(landmark,)):
                    cached[landmark]=points[landmark]
                    render[landmark]=points[landmark]
                    last_point[landmark]=index
                    point_status.append('measured')
                else:
                    point_status.append('held' if index-last_point[landmark]<=maximum else 'unavailable')
            current=calculator.calculate(points,self.sides)
            if self.options.view=='Frontal':
                both=calculator.calculate(points,('left','right'))
                for side in ('left','right'):
                    if f'{side}_knee_deg' in both:
                        current[f'{side}_q_estimated_deg']=180-both[f'{side}_knee_deg']
            keys=[f'{side}_{joint}_deg' for side in self.sides for joint in ('hip','knee','ankle')]
            if self.options.view=='Frontal':
                keys.extend(f'{side}_q_estimated_deg' for side in ('left','right'))
            metrics,status={},{}
            for key in keys:
                if key in current:
                    cached_metrics[key]=current[key]
                    last_metric[key]=index
                    metrics[key],status[key]=current[key],'measured'
                elif key in cached_metrics and index-last_metric[key]<=maximum:
                    metrics[key],status[key]=cached_metrics[key],'held'
                else:
                    metrics[key],status[key]=None,'unavailable'
            frames.append(DisplayFrame(render,metrics,status,point_status,self.sides))
            if any(value!='measured' for value in status.values()):
                self.records.append(dict(frame=index,angles=status.copy(),
                    held_landmarks=[i for i,state in enumerate(point_status) if state=='held'],
                    unavailable_landmarks=[i for i,state in enumerate(point_status) if state=='unavailable']))
        return frames


def dashed_line(frame,a,b,color,thickness=2):
    import cv2
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    length=np.linalg.norm(b-a)
    if not np.isfinite(length) or length<1:
        return
    # Cap drawing work for landmarks outside the image.
    if max(np.max(np.abs(a)),np.max(np.abs(b)))>max(frame.shape[:2])*4:
        return
    for start in range(0,int(length),14):
        first=a+(b-a)*(start/length)
        second=a+(b-a)*(min(start+8,length)/length)
        cv2.line(frame,tuple(first.astype(int)),tuple(second.astype(int)),color,thickness,cv2.LINE_AA)


def draw_joint_labels(frame,points,metrics,options,status=None):
    import cv2
    from i18n import tr
    status=status or {}
    for number,(key,value) in enumerate(metrics.items()):
        if not key.endswith('_deg'):
            continue
        side,joint=key.split('_')[:2]
        if side not in LANDMARK_IDS or joint not in ('hip','knee','ankle','q'):
            continue
        vertex=LANDMARK_IDS[side][{'hip':'HIP','knee':'KNEE','ankle':'ANKLE','q':'KNEE'}[joint]]
        if points is not None and np.all(np.isfinite(points[vertex,:2])):
            x,y=map(int,points[vertex,:2])
            x+=10
            y+=30 if joint=='q' else -12
        else:
            x,y=20,80+number*32
        letter=('L' if side=='left' else 'R') if options.language=='en' else ('E' if side=='left' else 'D')
        state=status.get(key,'measured')
        numeric='--' if value is None or not np.isfinite(value) else f'{value:.1f}'
        suffix=' ~' if state=='held' else ''
        label=f'{letter} {tr(options.language,joint)}: {numeric} deg{suffix}'
        color=(30,230,255) if side=='left' else (255,170,30)
        if state=='held': color=(30,170,255)
        elif state=='unavailable': color=(190,190,190)
        size=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.75,2)[0]
        x=max(5,min(x,frame.shape[1]-size[0]-5))
        y=max(size[1]+5,min(y,frame.shape[0]-8))
        cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),5,cv2.LINE_AA)
        cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2,cv2.LINE_AA)
    if any(state != 'measured' for state in status.values()):
        cv2.putText(frame,tr(options.language,'display_quality_legend'),(14,frame.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,(30,190,255),2,cv2.LINE_AA)
    return frame
