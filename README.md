# Calcium-dynamics-tracer
Program that analyzes calcium activities within cells *in vitro* from time-series calcium imaging data
* The program was built during my industry experience.
* The program goes through time-series imaging data, and trace temporal intensity changes at regions of interest.



## Advantages of Calcium Dynamics Tracer
* The program is designed to be user-friendly.
* The program provides discretion to decide how deep the analysis users want.
* The program is designed to be less labor-intensive compared to other commercial software e.g., MetaMorph.
* The program does not require any pre-processing of time-series imaging data e.g., converting to 8 bit, making videos, changing brightness & contrast, etc. 


## Overview of Calcium Dynamics Tracer
The following sequences show the analysis process.


### Get information from users
<img src="https://github.com/kiyunleeeee/Calcium-dynamics-tracer/blob/bd1d1b7e10eea6b7485146ebdc1b284310cc04bd/user%20input.png" width="25%" height="25%">

### Load image sequence
<img src="https://github.com/kiyunleeeee/Calcium-dynamics-tracer/blob/4b5902db1350dc1871961e7e62546f7ab56f3ade/image%20sequence.png" width="50%" height="50%">


### Select cells and backgrounds by users
<img src="https://github.com/kiyunleeeee/Calcium-dynamics-tracer/blob/f134776f59c1d8763291324c5894f52e648b0f20/select%20background.png" width="25%" height="25%">



### Trace calcium dynamics and subtract backgrounds

$$ {\Delta F \over F_0} = {{F_n-F_0} \over F_0} $$

$$ F_n = F_{n, raw} - F_{background} $$

$$ F_0 = F_{0, raw} - F_{background} $$

$F_n$: intensity at the $n^{th}$ frame after background subtraction

$F_0$: intensity at the resting phasese after background subtraction

<img src="https://github.com/kiyunleeeee/Calcium-dynamics-tracer/blob/48b3a2d4fee8cd77407a12ed7b397e937c88db20/calcium%20dynamics%20pattern.png" width="25%" height="25%">

Scale bars: 2 ${\Delta F \over F_0}$, 180 s




## Additional programs, Calcium Activity Pattern Generator & Dose Response Curve Generator
* The program contains additional programs, one generating calcium activity patterns and the other generating dose response curves.
* These are optional to use.


### Calcium Activity Pattern Generator
<img src="https://github.com/kiyunleeeee/Calcium-dynamics-tracer/blob/1f46a2619d72429f88f0021c0b24da062bdb9759/calcium%20dynamics%20patterns.png" width="25%" height="25%">

Scale bars: 10 ${\Delta F \over F_0}$, 180 s

### Dose Response Curve Generator
<img src="https://github.com/kiyunleeeee/Calcium-dynamics-tracer/blob/c0c274b2d5d165e842fcc8a1edda4e50f4552456/dose%20response%20curve.png" width="25%" height="25%">
