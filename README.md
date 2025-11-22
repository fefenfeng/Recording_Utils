# Recording_Utils

**录音及录音会话** --- 相关工具脚本及模块函数🧰



## Prerequisites

- python ~= 3.10.0
- 其他库及版本依赖见`requirements.txt`



## Documentation

### restructure_au_recordings

- **功能：** 按照记录的participant/session对应后缀的记录excel表格文档，整理AU录制的音频文件到三层目录结构

- **使用：** `python restructure_au_recordings.py`

  - 默认使用同目录下的`config.cfg`, 参数详情见注释

  - 数据记录excel表格形式如下:

  - | participant | 1st  | 2nd  | 3rd  |
    | :---------- | ---- | ---- | ---- |
    | 016041      | 001  | 002  | 003  |
    | 013330      | 004  | 006  | 007  |

- ⚠ 仅适用于三层目录结构`root/participant/session/*.wav`



### mic_sensitivity_calibration

#### compute_mic_sensitivity

- **功能:**  计算麦克风低频灵敏度差异 (**测试方式：**麦克悬挂在一个平面，远场(>1.5m)音响播放噪声)
- **使用：** `python compute_mic_sensitivity.py`
  - 默认使用同目录下的`config.cfg`, 参数详情见注释
- **输出：** 输出文件格式为`json`, 包含`session_level`, `distance_level`,`global_level`的mic通道对于参考mic通道的相对灵敏度差异
- ⚠ 仅适用于三层目录结构`root/distance/session/*.wav`

#### calibrate_mic_sensitivity

- **功能：** 根据麦克风灵敏度计算结果`json`文件对音频文件进行校准
- **使用：** `python calibrate_mic_sensitivity.py`， **使用前必须有相应麦克灵敏度计算结果`json`文件**
  - 默认使用同目录下的`config.cfg`，参数详情见注释

- ⚠ 仅适用于三层目录结构`root/distance/session/*.wav`



### find_minmax_wav

- **功能：** 深度遍历给定路径的目录下的所有`wav`文件，找出最长及最短的`wav`文件以及对应时间
- **使用：** `python find_minmax_wav.py <path>`



### mic_lowfreq_repair

- **功能：** 适用于临时补偿，因收音孔堵塞或脏污，导致mic通道已经录制的数据低频塌陷
- **使用：** `python mic_lowfreq_repair.py`
- ⚠️ 注意，当遇到mic低频明显出问题的情况时，需在下次实验/测试前换掉。此代码仅为补偿已经“低频塌陷”的mic。通过将一参考mic和损坏mic悬挂在距远场声源同距离的面上，播放稳态噪声，类似逐点灵敏度的方式进行校准