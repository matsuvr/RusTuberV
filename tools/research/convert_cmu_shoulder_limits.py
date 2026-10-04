#!/usr/bin/env python3
"""Fit full-rotation shoulder k-DOPs from explicitly selected CMU ROM clips.

Offline only; numpy is needed by this conversion, not by RusTuberV.
Input ASF/AMC files stay outside the repository. CMU permits copying/modifying/
redistributing its motion data and inclusion in commercial products, but not
resale of the dataset itself. See the upper-limb ADR and THIRD_PARTY_NOTICES.
No LAFAN1/PosePrior data and no arbitrary biological coefficients are used.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

CLIPS = [(74, 13), (126, 14), (127, 2), (143, 21)]
# Holden's full-rotation 26-DOP. Normalization changes units, not the fit.
DIRECTIONS = [(1,0,0),(0,1,0),(0,0,1),(1,1,1),(-1,1,1),(-1,-1,1),
              (1,-1,1),(0,1,1),(0,-1,1),(1,0,1),(-1,0,1),(1,1,0),(-1,1,0)]

def euler(values):
    x, y, z = np.deg2rad(values)
    sx,cx=np.sin(x),np.cos(x); sy,cy=np.sin(y),np.cos(y); sz,cz=np.sin(z),np.cos(z)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]) @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])

def unit(v):
    return v / np.linalg.norm(v)

def scaled_axis(m):
    # Davenport's symmetric matrix yields quaternion XYZW; choose the T-pose
    # hemisphere explicitly. No swing/twist separation discards correlations.
    k=np.array([
        [m[0,0]-m[1,1]-m[2,2], m[1,0]+m[0,1], m[2,0]+m[0,2], m[2,1]-m[1,2]],
        [m[1,0]+m[0,1],m[1,1]-m[0,0]-m[2,2],m[2,1]+m[1,2],m[0,2]-m[2,0]],
        [m[2,0]+m[0,2],m[2,1]+m[1,2],m[2,2]-m[0,0]-m[1,1],m[1,0]-m[0,1]],
        [m[2,1]-m[1,2],m[0,2]-m[2,0],m[1,0]-m[0,1],np.trace(m)]])
    _, vectors=np.linalg.eigh(k)
    q=vectors[:,-1]
    if q[3]<0: q=-q
    n=np.linalg.norm(q[:3])
    return q[:3]*(2*np.arctan2(n,q[3])/n) if n>1e-12 else 2*q[:3]

def bones(path):
    result={}; current=None; active=False
    for line in path.read_text().splitlines():
        words=line.split()
        if not words: continue
        if words[0]==':bonedata': active=True
        if words[0]==':hierarchy': break
        if not active: continue
        if words[0]=='begin': current={}
        elif words[0]=='end': result[current['name']]=current
        elif current is not None and words[0] in ('name','axis','direction','dof'):
            key=words[0]; data=words[1:]
            current[key]=data[0] if key=='name' else (data if key=='dof' else np.array(list(map(float,data[:3]))))
    return result

def samples(asf,amc):
    rig=bones(asf); transforms={}
    for side in ('l','r'):
        u=unit(rig[side+'humerus']['direction'])
        h=euler(rig[side+'radius']['axis'])[:,0]
        h=unit(h-u*np.dot(h,u))
        source=np.column_stack((u,h,np.cross(u,h)))
        sign=1 if side=='l' else -1
        du=np.array([sign,0.,0.]); dh=np.array([0.,-sign,0.])
        # Measured source elbow-axis orientation defines the anatomical T
        # reference; it is not assumed to be the ASF zero Euler orientation.
        neutral=np.column_stack((du,dh,np.cross(du,dh))) @ source.T
        c=euler(rig[side+'humerus']['axis'])
        transforms[side]=(c,neutral)
    data=[]
    for line in amc.read_text().splitlines():
        words=line.split()
        if not words or words[0] not in ('lhumerus','rhumerus'): continue
        side=words[0][0]; c,neutral=transforms[side]
        q=c @ euler(list(map(float,words[1:]))) @ c.T @ neutral.T
        if side=='l':
            reflection=np.diag([-1,1,1]); q=reflection @ q @ reflection
        data.append(scaled_axis(q))
    return np.array(data)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    all_samples=[]; inputs=[]
    for subject,trial in CLIPS:
        asf=args.directory/f'{subject:02}.asf'; amc=args.directory/f'{subject:02}_{trial:02}.amc'
        values=samples(asf,amc); all_samples.extend(values)
        inputs.append({'subject':subject,'trial':trial,'side_samples':len(values),
                       'asf_sha256':hashlib.sha256(asf.read_bytes()).hexdigest(),
                       'amc_sha256':hashlib.sha256(amc.read_bytes()).hexdigest()})
    cloud=np.array(all_samples)
    axes=np.array([unit(np.array(v,dtype=float)) for v in DIRECTIONS])
    projected=cloud @ axes.T
    output={'format':1,'source':'CMU Graphics Lab Motion Capture Database',
            'coordinates':'right anatomical T-reference, GH full quaternion principal scaled axis, radians',
            'inputs':inputs,'axes':axes.tolist(),'min':projected.min(axis=0).tolist(),'max':projected.max(axis=0).tolist()}
    args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    def rust_float(value):
        if abs(abs(value) - np.sqrt(0.5)) < 1e-12:
            return ('-' if value < 0 else '') + 'std::f32::consts::FRAC_1_SQRT_2'
        return str(np.float32(value))
    rows=[]
    for axis,low,high in zip(axes,projected.min(axis=0),projected.max(axis=0)):
        rows.append('    ['+', '.join(rust_float(v) for v in (*axis,low,high))+'],')
    args.output.with_suffix('.rs').write_text(
        '// Generated by tools/research/convert_cmu_shoulder_limits.py.\n'
        '// CMU data provenance and hashes: cmu_shoulder_rom.json.\n'
        'pub(super) const SHOULDER_ROM: [[f32; 5]; 13] = [\n'+'\n'.join(rows)+'\n];\n',encoding='utf-8')
    print(f'{len(cloud)} full-rotation side samples -> {args.output}')

if __name__=='__main__': main()
