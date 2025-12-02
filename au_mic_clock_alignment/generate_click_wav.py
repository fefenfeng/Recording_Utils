# -*- coding: utf-8 -*-
"""
@Author: Yuheng Feng

@Date: 2025/11/28 10:56

@Description: 生成click tag wav文件
"""
import numpy as np
from scipy.io.wavfile import write


sr = 48000
click = np.zeros(sr)
click[100] = 1.0
write("click.wav", sr, click.astype(np.float32))
