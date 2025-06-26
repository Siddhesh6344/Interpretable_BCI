# Motion correction parameters
coreParams = {
    'motionCorrection': True,
    'method': 'normcorre'
}

if coreParams['motionCorrection']:
    from correctMotion_3D import correct_motion_3d
    iDop, coreParams = correct_motion_3d(iDop, coreParams)
else:
    coreParams['method'] = 'N/A'
