from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import threading
import time
import unittest

import json
import os

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"python"))
from robo_lab.inference.config import load_config
from robo_lab.inference.kinematics import Kinematics, rotation_error
from robo_lab.inference.observation import ObservationStore, CAMERAS, decode_rgb
from robo_lab.inference.policy import CirclePolicy, validate_actions
from robo_lab.inference.lingbot_policy import lingbot_action, lingbot_observation
from robo_lab.inference.worker import resolve_policy
from robo_lab.inference.executor import Executor


def config():
    return load_config(ROOT/"config/inference/python_circle.yaml")


def observation(cfg):
    return {"image": {k: np.zeros((h,w,3), np.uint8) for k,(h,w,_,_) in CAMERAS.items()},
            "state": np.r_[cfg["robot_config"]["arm_home"], 0.25], "prompt": cfg["prompt"]}


def local_endpoint():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return f'tcp/127.0.0.1:{sock.getsockname()[1]}'


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.kin = Kinematics(self.cfg["kinematics_config"])
        self.obs = observation(self.cfg)

    def test_color_and_stride(self):
        for name,(h,w,c,fmt) in CAMERAS.items():
            raw = np.zeros((h,w*c+8), np.uint8)
            raw[:, :w*c] = np.tile(np.arange(c, dtype=np.uint8)+10, w)
            rgb = decode_rgb(name, raw.tobytes(), w,h,c,fmt,w*c+8)
            np.testing.assert_array_equal(rgb[0,0], [12,11,10] if c==4 else [10,11,12])
            self.assertEqual(rgb.shape,(h,w,3))
            with self.assertRaises(ValueError):
                decode_rgb(name,b'bad',w,h,c,fmt)

    def test_jacobian_finite_difference(self):
        for offset in (0., 0.03, -0.04):
            q = self.obs["state"][:7]+offset
            j = self.kin.jacobian(q)
            base = self.kin.fk(q)
            for i in range(7):
                qt=q.copy(); qt[i]+=1e-6
                t=self.kin.fk(qt)
                numeric = np.r_[(t[:3,3]-base[:3,3])/1e-6,
                                 rotation_error(t[:3,:3],base[:3,:3])/1e-6]
                np.testing.assert_allclose(j[:,i],numeric,atol=1e-6)

    def test_cpp_fk_parity(self):
        # Compile the repository's actual C++ implementation, not a second copy.
        source = r'''
#include "cartesian_control/jacobian_calculator.h"
#include <iostream>
#include <iomanip>
int main(){
 robo_lab::JacobianCalculator k; std::vector<std::array<double,4>> dh(7);
 for(auto& row:dh) for(auto& x:row) std::cin>>x; k.set_dh_params(dh);
 Eigen::Matrix4d tool; for(int i=0;i<4;i++) for(int j=0;j<4;j++) std::cin>>tool(i,j);
 std::vector<double> q(7); for(auto& x:q) std::cin>>x;
 auto t=k.compute_ee_transform(q,tool); auto jac=k.compute_jacobian(q,tool);
 std::cout<<std::setprecision(17);
 for(int i=0;i<4;i++) for(int j=0;j<4;j++) std::cout<<t(i,j)<<" ";
 for(int i=0;i<6;i++) for(int j=0;j<7;j++) std::cout<<jac(i,j)<<" ";
}
'''
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); (p/'check.cpp').write_text(source)
            subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT/'include'),'-I/usr/include/eigen3',
                            str(p/'check.cpp'),str(ROOT/'src/cartesian_control/jacobian_calculator.cpp'),
                            '-o',str(p/'check')],check=True,capture_output=True)
            for offset in (0., 0.03):
                q=self.obs['state'][:7]+offset
                values=np.r_[self.kin.dh.ravel(),self.kin.tool.ravel(),q]
                result=subprocess.check_output([str(p/'check')],input=' '.join(map(str,values)),text=True)
                a=np.fromstring(result,sep=' ')
                np.testing.assert_allclose(a[:16].reshape(4,4),self.kin.fk(q),atol=1e-12)
                np.testing.assert_allclose(a[16:].reshape(6,7),self.kin.jacobian(q),atol=1e-12)

    def test_circle_chunks_hold_reset(self):
        p=CirclePolicy(self.cfg,self.kin)
        chunks=[p.infer(self.obs) for _ in range(7)]
        a=np.concatenate([c['action'] for c in chunks])
        xyz=np.concatenate([c['target_xyz'] for c in chunks])
        self.assertEqual(a.shape,(350,8))
        np.testing.assert_allclose(a[:,7],0.25)
        self.assertLess(np.abs(np.diff(a[:,:7],axis=0)).max(),self.cfg['max_joint_speed']/30)
        for command,goal in zip(a,xyz):
            t=self.kin.fk(command[:7])
            self.assertLess(np.linalg.norm(t[:3,3]-goal),0.001)
            self.assertLess(np.linalg.norm(rotation_error(p.start[:3,:3],t[:3,:3])),np.deg2rad(.5))
        np.testing.assert_allclose(xyz[299],p.start[:3,3],atol=1e-12)
        np.testing.assert_allclose(a[300:],np.tile(a[299],(50,1)),atol=1e-12)
        p.reset()
        np.testing.assert_array_equal(p.infer(self.obs)['action'],chunks[0]['action'])

    def test_bad_actions_and_checkpoint(self):
        baseline=np.tile(self.obs['state'],(50,1))
        for mutation in ('nan','limit','gripper','jump','shape'):
            a=baseline.copy()
            if mutation=='nan': a[-1,0]=np.nan
            if mutation=='limit': a[-1,0]=4
            if mutation=='gripper': a[-1,7]=1.01
            if mutation=='jump': a[-1,0]+=.02
            if mutation=='shape': a=a[:49]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_actions(a,self.obs['state'],self.cfg,self.kin)
        import yaml
        original=yaml.safe_load((ROOT/'config/inference/python_circle.yaml').read_text())
        original['policy']='checkpoint'
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.yaml'; path.write_text(yaml.safe_dump(original))
            with self.assertRaisesRegex(ValueError,'checkpoint_path'): load_config(path)
            original['checkpoint_path']='not_ready.ckpt'; path.write_text(yaml.safe_dump(original))
            with self.assertRaises(NotImplementedError): load_config(path)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.cfg=config(); self.kin=Kinematics(self.cfg['kinematics_config'])
        self.obs=observation(self.cfg); self.now=0.; self.sent=[]; self.calls=0
        self.store=ObservationStore(self.cfg['prompt'],200)
        outer=self
        class Hold:
            def reset(self): pass
            def infer(self,obs):
                outer.calls+=1
                return {'action': np.tile(obs['state'],(50,1))}
        self.policy=Hold()
        self.e=Executor(self.cfg,self.kin,self.policy,self.store,
                        lambda a,s:self.sent.append((self.now,a)),clock=lambda:self.now)
        self.refresh()

    def refresh(self):
        for k,v in self.obs['image'].items(): self.store.put(k,v,self.now)
        self.store.put('state',self.obs['state'],self.now)

    def test_chunk_frequency_no_catchup(self):
        self.e.set_state('inference')
        for i in range(50):
            self.now=i/30+1e-9*i; self.refresh(); self.assertTrue(self.e.tick())
        self.assertEqual(self.calls,1)
        self.now+=1/30+1e-9; self.refresh(); self.e.tick()
        self.assertEqual(self.calls,2)
        self.now+=.15; self.refresh(); self.assertTrue(self.e.tick())
        for _ in range(10): self.assertFalse(self.e.tick())

    def test_stale_latched_and_reentry_home(self):
        self.e.set_state('inference'); self.assertTrue(self.e.tick())
        sent=len(self.sent)
        self.now=.201
        self.assertFalse(self.e.tick())
        self.assertIsNone(self.e.fault)
        self.assertEqual(len(self.sent),sent)
        self.assertIsNotNone(self.e.chunk)
        self.now+=1/30; self.refresh()
        self.assertTrue(self.e.tick())
        self.assertIsNone(self.e.fault)
        self.now+=.201
        self.assertFalse(self.e.tick())
        self.assertIsNone(self.e.fault)
        self.now+=self.cfg['stale_fault_s']+1e-6
        self.assertFalse(self.e.tick())
        self.assertIn('Stale',self.e.fault)
        self.refresh(); self.assertFalse(self.e.tick())
        self.e.set_state('standby'); self.e.set_state('inference')
        self.obs['state'][0]+=.1; self.refresh()
        self.assertFalse(self.e.tick()); self.assertIn('Not at home',self.e.fault)

    def test_invalid_chunk_and_state_transition_during_infer(self):
        self.e.set_state('inference')
        self.policy.infer=lambda obs:{'action':np.full((50,8),np.nan)}
        self.assertFalse(self.e.tick()); self.assertEqual(len(self.sent),0)
        self.e.set_state('standby'); self.e.set_state('inference')
        def changing(obs):
            self.e.set_state('standby'); self.e.set_state('inference')
            return {'action':np.tile(obs['state'],(50,1))}
        self.policy.infer=changing
        self.assertFalse(self.e.tick()); self.assertEqual(len(self.sent),0)

    def test_sensor_expiry_during_infer(self):
        self.e.set_state('inference')
        def slow(obs):
            self.now+=.21
            return {'action':np.tile(obs['state'],(50,1))}
        self.policy.infer=slow
        self.assertFalse(self.e.tick()); self.assertEqual(self.sent,[])
        self.assertIsNone(self.e.fault)
        self.assertIsNotNone(self.e.chunk)
        self.now+=self.cfg['stale_fault_s']+1e-6
        self.assertFalse(self.e.tick())
        self.assertIn('Stale',self.e.fault)


class ZenohTests(unittest.TestCase):
    def test_streaming_circle_and_cadence(self):
        import zenoh
        from robo_lab.inference.transport import ZenohNode, zenoh_config
        from robo_lab.inference.proto import franka_pb2,kinect_pb2
        cfg=config(); obs=observation(cfg); stop=threading.Event()
        events=[]; received=[]
        endpoints=[local_endpoint()]
        with zenoh.open(zenoh_config(listen=endpoints,isolated=True)) as host:
            with zenoh.open(zenoh_config(connect=endpoints,isolated=True)) as client:
                node=ZenohNode(cfg,client,lambda **kw:events.append(kw))
                sub=host.declare_subscriber(cfg['topics']['command'],lambda s:received.append(
                    (time.monotonic(),franka_pb2.RobotCommand.FromString(s.payload.to_bytes()))))
                messages={}
                for key,(h,w,c,fmt) in CAMERAS.items():
                    messages[key]=kinect_pb2.rgbImage(width=w,height=h,channels=c,type=fmt,step=w*c,
                        image=np.zeros((h,w,c),np.uint8).tobytes()).SerializeToString()
                def stream():
                    due={k:0. for k in CAMERAS}
                    while not stop.is_set():
                        now=time.monotonic()
                        for key,hz in (('kinec2',20),('realsense',15)):
                            if now>=due[key]:
                                host.put(cfg['topics'][key],messages[key]); due[key]=now+1/hz
                        state=franka_pb2.RobotObservation()
                        values=([j.position for j in received[-1][1].joints] if received else obs['state'])
                        for value in values: state.joints.add(position=value)
                        host.put(cfg['topics']['state'],state.SerializeToString())
                        stop.wait(.01)
                time.sleep(.2)
                worker=threading.Thread(target=stream); worker.start()
                try:
                    deadline=time.monotonic()+2
                    while len(node.store.values)<3 and time.monotonic()<deadline: time.sleep(.005)
                    host.put(cfg['topics']['state_machine'],b'inference')
                    deadline=time.monotonic()+5
                    while len(received)<65 and time.monotonic()<deadline:
                        node.executor.tick(); time.sleep(.001)
                    self.assertIsNone(node.executor.fault)
                    self.assertGreaterEqual(len(received),65)
                    commands=[e for e in events if e['event']=='command']
                    infers=[e for e in events if e['event']=='infer']
                    self.assertEqual(len(infers),2)
                    self.assertEqual([c['index'] for c in commands[:50]],list(range(50)))
                    intervals=np.diff([c['published_monotonic'] for c in commands])
                    self.assertGreaterEqual(intervals.min(),1/30-1e-5)
                    self.assertLess(np.median(intervals),.045)
                    a=np.array([[j.position for j in m.joints] for _,m in received])
                    self.assertGreater(np.ptp(a[:,0]),1e-5)
                    host.put(cfg['topics']['state_machine'],b'standby')
                    deadline=time.monotonic()+1
                    while node.executor.state!='standby' and time.monotonic()<deadline: time.sleep(.005)
                    count=len(received)
                    for _ in range(10): node.executor.tick(); time.sleep(.005)
                    self.assertEqual(len(received),count)
                finally:
                    stop.set(); worker.join(timeout=2); node.close(); sub.undeclare()

    def test_isolated_wire_and_fault(self):
        import zenoh
        from robo_lab.inference.transport import ZenohNode, zenoh_config
        from robo_lab.inference.proto import franka_pb2,kinect_pb2
        cfg=config(); obs=observation(cfg)
        # Both peers explicitly disable discovery; no production topics escape localhost.
        endpoints=[local_endpoint()]
        with zenoh.open(zenoh_config(listen=endpoints,isolated=True)) as host:
            with zenoh.open(zenoh_config(connect=endpoints,isolated=True)) as client:
                node=ZenohNode(cfg,client)
                received=[]
                sub=host.declare_subscriber(cfg['topics']['command'],lambda s:received.append(
                    franka_pb2.RobotCommand.FromString(s.payload.to_bytes())))
                time.sleep(.3)
                for key,(h,w,c,fmt) in CAMERAS.items():
                    msg=kinect_pb2.rgbImage(width=w,height=h,channels=c,type=fmt,step=w*c,
                                           image=np.zeros((h,w,c),np.uint8).tobytes())
                    host.put(cfg['topics'][key],msg.SerializeToString())
                msg=franka_pb2.RobotObservation()
                for v in obs['state']: msg.joints.add(position=v)
                host.put(cfg['topics']['state'],msg.SerializeToString())
                deadline=time.monotonic()+2
                while len(node.store.values)<3 and time.monotonic()<deadline: time.sleep(.005)
                host.put(cfg['topics']['state_machine'],b'inference')
                while node.executor.state!='inference' and time.monotonic()<deadline: time.sleep(.005)
                # Hold backend isolates transport from solver timing.
                node.executor.policy.infer=lambda o:{'action':np.tile(o['state'],(50,1))}
                self.assertTrue(node.executor.tick(),node.executor.fault)
                deadline=time.monotonic()+1
                while not received and time.monotonic()<deadline: time.sleep(.005)
                self.assertEqual(len(received),1)
                self.assertEqual(received[0].mode,'position')
                self.assertEqual(received[0].type,franka_pb2.RobotCommand.TYPE_JOINT_TARGET)
                np.testing.assert_allclose([j.position for j in received[0].joints],obs['state'])
                time.sleep(.21)
                self.assertFalse(node.executor.tick())
                self.assertIsNone(node.executor.fault)
                self.assertEqual(len(received),1)
                time.sleep(cfg['stale_fault_s']+.05)
                self.assertFalse(node.executor.tick())
                self.assertIn('Stale',node.executor.fault)
                host.put(cfg['topics']['state_machine'],b'standby')
                deadline=time.monotonic()+1
                while node.executor.state!='standby' and time.monotonic()<deadline: time.sleep(.005)
                self.assertFalse(node.executor.tick())
                self.assertIsNone(node.executor.chunk)
                node.close(); sub.undeclare()


class LingbotTests(unittest.TestCase):
    def test_observation_and_action_contract(self):
        from robo_lab.inference.lingbot_policy import lingbot_action, lingbot_observation
        from robo_lab.inference.worker import resolve_policy
        cfg=config(); obs=observation(cfg)
        mapped=lingbot_observation(obs)
        self.assertEqual(mapped['task'], cfg['prompt'])
        np.testing.assert_array_equal(mapped['observation.state'], obs['state'].astype(np.float32))
        self.assertIs(mapped['observation.images.kinec2'], obs['image']['kinec2'])
        self.assertIs(mapped['observation.images.realsense'], obs['image']['realsense'])
        action=np.tile(obs['state'], (50, 1))
        packed=lingbot_action({'action': action.astype(np.float32)}, 50)
        self.assertEqual(packed['action'].dtype, np.float64)
        self.assertEqual(packed['action'].shape, (50, 8))
        with self.assertRaises(ValueError): lingbot_action({'action': action[:49]}, 50)
        with self.assertRaises(ValueError): lingbot_action({}, 50)
        self.assertIs(resolve_policy(cfg), CirclePolicy)
        self.assertEqual(resolve_policy({'policy': 'lingbot_vla_v2'}).__name__, 'LingbotPolicy')

    def test_franka_training_files(self):
        import json
        import yaml
        robot=yaml.safe_load((ROOT/'config/inference/franka_real.yaml').read_text())
        arm=robot['actions'][0]['action.arm.position']
        grip=robot['actions'][1]['action.effector.position']
        self.assertTrue(arm['subtract_state'])
        self.assertFalse(grip['subtract_state'])
        self.assertEqual(arm['origin_keys'][0]['action'], {'start': 0, 'end': 7})
        self.assertEqual(grip['origin_keys'][0]['action'], {'start': 7, 'end': 8})
        self.assertEqual(robot['states'][0]['observation.state.arm.position']['origin_keys'][0]['observation.state'],
                         {'start': 0, 'end': 7})
        self.assertEqual(robot['images'][0]['observation.images.camera_top']['origin_keys'],
                         'observation.images.kinec2')
        self.assertEqual(robot['images'][1]['observation.images.camera_wrist_left']['origin_keys'],
                         'observation.images.realsense')
        stats=json.loads((ROOT/'config/inference/franka_real_norm_stats.json').read_text())
        self.assertEqual(stats['count'], 32912)
        self.assertEqual(len(stats['norm_stats']['action.arm.position']['mean']), 7)
        self.assertEqual(len(stats['norm_stats']['action.effector.position']['mean']), 1)
        self.assertEqual(len(stats['norm_stats']['observation.state.arm.position']['std']), 7)
        cli=yaml.safe_load((ROOT/'config/inference/franka_lingbotvla_cli.yaml').read_text())
        self.assertEqual(cli['data']['data_name'], 'franka_real')
        self.assertEqual(cli['data']['cameras'][:2], ['camera_top', 'camera_wrist_left'])
        self.assertEqual(cli['data']['img_size'], 256)
        self.assertEqual(cli['data']['state_norm_type'], 'none')
        self.assertEqual(cli['data']['norm_type'][0], "{'arm.position': 'meanstd'}")
        self.assertEqual(cli['train']['chunk_size'], 50)
        self.assertEqual(cli['train']['max_action_dim'], 55)
        self.assertEqual(cli['train']['align_params']['mode'], 'query')
        shipped=yaml.safe_load((ROOT/'config/inference/python_lingbot.yaml').read_text())
        self.assertEqual(shipped['policy'], 'lingbot_vla_v2')
        self.assertEqual(shipped['prompt'], 'franka manipulation')
        self.assertEqual(shipped['inference_timeout_s'], 60.0)
        self.assertEqual(shipped['policy_init_timeout_s'], 300)
        self.assertEqual(shipped['chunk_size'], 50)

    def test_load_config_requires_qwen_and_complete_shards(self):
        import json
        import os
        import yaml
        try:
            loaded=load_config(ROOT/'config/inference/python_lingbot.yaml')
        except ValueError as exc:
            self.assertTrue(any(token in str(exc) for token in (
                'incomplete', 'QWEN3VL_PATH', 'missing', 'shards')), str(exc))
        else:
            self.assertEqual(loaded['policy'], 'lingbot_vla_v2')
            self.assertTrue(Path(loaded['qwen3vl_path']).is_dir())
        base=yaml.safe_load((ROOT/'config/inference/python_lingbot.yaml').read_text())
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            ckpt=temp/'ckpt'
            ckpt.mkdir()
            (ckpt/'model.safetensors').write_bytes(b'not-a-real-shard')
            (ckpt/'model.safetensors.index.json').write_text(
                json.dumps({'weight_map': {'weight': 'model.safetensors'}}))
            qwen=temp/'qwen'
            qwen.mkdir()
            base['checkpoint_path']=str(ckpt)
            path=temp/'inference.yaml'
            path.write_text(yaml.safe_dump(base))
            old=os.environ.pop('QWEN3VL_PATH', None)
            try:
                with self.assertRaisesRegex(ValueError, 'QWEN3VL_PATH'):
                    load_config(path)
                os.environ['QWEN3VL_PATH']=str(qwen)
                cfg=load_config(path)
                self.assertEqual(cfg['qwen3vl_path'], str(qwen.resolve()))
                self.assertTrue(cfg['lingbot_robot_config'].endswith('franka_real.yaml'))
                self.assertTrue(cfg['norm_stats'].endswith('franka_real_norm_stats.json'))
                self.assertEqual(cfg['policy_init_timeout_s'], 300)
                (ckpt/'model-00004.safetensors.baiduyun.p.downloading').write_bytes(b'')
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    load_config(path)
            finally:
                if old is None: os.environ.pop('QWEN3VL_PATH', None)
                else: os.environ['QWEN3VL_PATH']=old


if __name__=='__main__': unittest.main()
