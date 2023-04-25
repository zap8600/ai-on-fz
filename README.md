# ai-on-fz
An attempt to get TensorFlow Lite for Microcontrollers running on the Flipper Zero.

This currently doesn't work! In order to get anywhere, the `cc.scons` file in fbt's `site_scons` folder must be modified to add multiple `-Wno-` arguments, which causes a linker error involving `--gc-sections`. 

## Installation
Run `pip3 install -r requirements.txt` to install `ufbt`, which lets you build the app without needing to clone the firmware repo.

## Progression
In order for the library to work, the linker error should need to be traced. If anyone has any suggestions on how to fix this, it would be greatly appreciated.
