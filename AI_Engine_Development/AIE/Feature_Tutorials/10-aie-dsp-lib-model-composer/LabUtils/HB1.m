%
% Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
% SPDX-License-Identifier: X11
%

function Hd = HB1
%HB1 Returns a discrete-time filter object.

% MATLAB Code
% Generated by MATLAB(R) 9.7 and DSP System Toolbox 9.9.
% Generated on: 23-Nov-2020 15:30:12

% Equiripple Halfband lowpass filter designed using the FIRHALFBAND
% function.

% All frequency values are normalized to 1.

Fpass = 0.1;               % Passband Frequency
Dpass = 5.7564627261e-05;  % Passband Ripple

% Calculate the coefficients using the FIRPM function.
b  = firhalfband('minorder', Fpass, Dpass);
Hd = dfilt.dffir(b);

% [EOF]
