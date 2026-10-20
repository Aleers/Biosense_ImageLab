import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from gait_events import FootfallRegistry, GaitPhaseAnalyzer
from processor import Analysis, ProcessingOptions, draw_overlay, draw_angle_table
from visual_continuity import VisualContinuity, draw_joint_labels
from test_robustness import body


def phases_for(size, contacts):
    phases={side:dict(event=['']*size,valid=np.ones(size,dtype=bool),stance=np.ones(size,dtype=bool))
            for side in ('left','right')}
    for index,side in contacts:
        phases[side]['event'][index]='CONTATO INICIAL'
    return phases

class FootfallTests(unittest.TestCase):
    def registry(self,size=40,calibration=True):
        data=Analysis(Path('video'),30,640,480,[body() for _ in range(size)])
        if calibration:
            data.calibration_points=((0,0),(100,0))
            data.calibration_cm=10
        return data,FootfallRegistry(data)

    def test_alternation_independent_of_distance(self):
        data,registry=self.registry()
        phases=phases_for(40,[(5,'left'),(15,'right'),(25,'left')])
        for i,points in enumerate(data.landmarks):
            points[27,0]=100
            points[28,0]=101  # 0.1 cm; the old 5 cm rule dropped the contact.
            registry.update(i,points,phases)
        self.assertEqual([entry['side'] for entry in registry.footfalls],['left','right','left'])
        self.assertTrue(all(entry['step_cm']<5 for entry in registry.footfalls))

    def test_uncalibrated_contacts_are_numbered(self):
        data,registry=self.registry(calibration=False)
        phases=phases_for(40,[(5,'left'),(15,'right')])
        rows=[registry.update(i,p,phases) for i,p in enumerate(data.landmarks)]
        self.assertEqual(rows[5]['step_number'],1)
        self.assertEqual(rows[15]['step_number'],2)
        self.assertTrue(all(entry['step_cm'] is None for entry in registry.footfalls))

    def test_missing_ankle_does_not_lose_contact(self):
        data,registry=self.registry()
        phases=phases_for(40,[(5,'left')])
        data.landmarks[5][28,2]=0
        for i,points in enumerate(data.landmarks[:8]):
            registry.update(i,points,phases)
        self.assertEqual(len(registry.footfalls),1)
        self.assertEqual(registry.footfalls[0]['frame'],5)
        self.assertEqual(registry.footfalls[0]['measurement_frame'],6)
        self.assertIsNotNone(registry.footfalls[0]['step_cm'])

    def test_nonalternating_candidate_is_audited_without_blocking_next(self):
        data,registry=self.registry()
        phases=phases_for(40,[(5,'left'),(15,'left'),(25,'right')])
        for i,points in enumerate(data.landmarks):
            registry.update(i,points,phases)
        self.assertEqual([entry['side'] for entry in registry.footfalls],['left','right'])
        self.assertEqual(registry.decisions[1]['reason'],'nonalternating_contact')
        self.assertFalse(registry.decisions[1]['accepted'])

    def test_initial_support_is_not_a_new_footfall(self):
        data,registry=self.registry()
        phases=phases_for(40,[])
        registry.update(0,data.landmarks[0],phases)
        self.assertFalse(registry.support_confirmed['left'])
        self.assertEqual(registry.footfalls,[])
        phases['left']['event'][5]='CONTATO INICIAL'
        registry.update(5,data.landmarks[5],phases)
        self.assertTrue(registry.support_confirmed['left'])
        self.assertEqual(len(registry.footfalls),1)

class PhaseTests(unittest.TestCase):
    def sequence(self):
        points=body()
        points[[29,30,31,32],1]=450
        return [points.copy() for _ in range(30)]

    def test_single_frame_flicker_does_not_add_contacts(self):
        sequence=self.sequence()
        for index in (6,12,18):
            sequence[index][[29,31],1]=380
        phases=GaitPhaseAnalyzer().analyze(Analysis(Path('video'),30,640,480,sequence))
        self.assertEqual(phases['left']['event'].count('CONTATO INICIAL'),0)

    def test_one_visible_foot_marker_is_enough(self):
        sequence=self.sequence()
        for points in sequence:
            points[29,2]=0
        phases=GaitPhaseAnalyzer().analyze(Analysis(Path('video'),30,640,480,sequence))
        self.assertTrue(phases['left']['valid'][10])

    def test_contact_requires_confirmed_swing_then_support(self):
        sequence=self.sequence()
        for index in range(8,16):
            sequence[index][[29,31],1]=380
        phases=GaitPhaseAnalyzer().analyze(Analysis(Path('video'),30,640,480,sequence))
        self.assertEqual(phases['left']['event'].count('CONTATO INICIAL'),1)
        self.assertEqual(phases['left']['event'].count('RETIRADA DO PE'),1)

    def test_long_gap_does_not_fabricate_contact(self):
        sequence=self.sequence()
        for index in range(6,14):
            sequence[index]=None
        phases=GaitPhaseAnalyzer().analyze(Analysis(Path('video'),30,640,480,sequence))
        self.assertEqual(phases['left']['event'].count('CONTATO INICIAL'),0)

class DisplayTests(unittest.TestCase):
    def test_short_hold_then_unavailable_without_changing_measurements(self):
        initial=body()
        low=initial.copy()
        low[25,2]=0.2
        frames=VisualContinuity(ProcessingOptions(side='Esquerdo')).build([initial]+[low]*10,30)
        self.assertEqual(frames[1].status['left_knee_deg'],'held')
        self.assertIsNone(frames[10].metrics['left_knee_deg'])
        self.assertEqual(frames[10].status['left_knee_deg'],'unavailable')
        self.assertTrue(np.all(np.isfinite(frames[10].points[25,:2])))
        image,measured=draw_overlay(np.zeros((480,640,3),np.uint8),low,ProcessingOptions(side='Esquerdo'),display=frames[1])
        self.assertNotIn('left_knee_deg',measured)
        self.assertEqual(low[25,2],np.float32(.2))
        self.assertGreater(int(image.sum()),0)

    def test_labels_remain_after_full_detection_loss(self):
        opts=ProcessingOptions(side='Esquerdo')
        frames=VisualContinuity(opts).build([body()]+[None]*10,30)
        image=np.zeros((480,640,3),np.uint8)
        with patch('cv2.putText',wraps=cv2.putText) as text:
            _,metrics=draw_overlay(image,None,opts,display=frames[-1])
        self.assertEqual(metrics,{})
        labels=[call.args[1] for call in text.call_args_list]
        self.assertTrue(any('Joelho: --' in label for label in labels))
        self.assertTrue(any('Quadril: --' in label for label in labels))
        self.assertTrue(any('Tornoz.: --' in label for label in labels))

    def test_joint_labels_are_clamped_inside_image(self):
        points=body()
        points[25,:2]=[630,2]
        image=np.zeros((480,640,3),np.uint8)
        with patch('cv2.putText',wraps=cv2.putText) as text:
            draw_joint_labels(image,points,{'left_knee_deg':120},ProcessingOptions(side='Esquerdo'))
        for call in text.call_args_list:
            x,y=call.args[2]
            self.assertGreaterEqual(x,5)
            self.assertGreaterEqual(y,5)
            width=cv2.getTextSize(call.args[1],cv2.FONT_HERSHEY_SIMPLEX,.75,2)[0][0]
            self.assertLessEqual(x+width,635)

    def test_angle_table_keeps_missing_rows(self):
        image=np.zeros((480,640,3),np.uint8)
        with patch('cv2.putText',wraps=cv2.putText) as text:
            draw_angle_table(image,{'left_knee_deg':None},{})
        self.assertTrue(any(call.args[1]=='E Joelho' for call in text.call_args_list))
        self.assertTrue(any(call.args[1]=='--' for call in text.call_args_list))

if __name__=='__main__':
    unittest.main()
