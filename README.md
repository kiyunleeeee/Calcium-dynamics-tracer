# Calcium-dynamics-tracer
Program that analyzes calcium activities within cells *in vitro* from time-series calcium imaging data
* The program was built during my industry experience.
* The program goes through time-series imaging data, and trace temporal intensity changes at regions of interest.



## Advantages of Calcium Dynamics Tracer
* The program is designed to be user-friendly.
* The program provides users discretion to decide how deep the analysis users want.
* The program is designed to be less labor-intensive compared to other commercial software e.g., MetaMorph
* The program does not require any pre-processing of time-series imaging data e.g., converting to 8 bit, making videos, changing brightness & contrast, etc. 


## Overview of Calcium Dynamics Tracer
The following sequences show the analysis process.


### Get information from users


### Load image sequence


### Select cells and backgrounds by users
Cell & background selection
![alt text](https://github.com/kiyunleeeee/Calcium-dynamics-tracer/blob/f134776f59c1d8763291324c5894f52e648b0f20/select%20background.png)



### Trace calcium dynamics and subtract backgrounds

$$ {\Delta F \over F_0} = {{F_n-F_0} \over F_0} $$

$$ F_n = F_{n, raw} - F_{background} $$

$$ F_0 = F_{0, raw} - F_{background} $$

$F_n$: intensity at the $n^{th}$ frame after background subtraction

$F_0$: intensity at the resting phasese after background subtraction



## Additional programs, Calcium Activity Pattern Generator and Dose Response Curve Generator
* The program contains additional programs, one generating calcium activity patterns and the other generating dose response curves.
* These are optional to use.


### Calcium Activity Pattern Generator


### Dose Response Curve Generator
