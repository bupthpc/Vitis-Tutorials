%
% Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
% SPDX-License-Identifier: MIT
%
% Author: Mark Rollins

clear all;
close all;
addpath('../../matlab');

% ------------------------------------------------------------
% Create Permutation Mapping
% ------------------------------------------------------------

N1 = 7;
N2 = 9; 
N3 = 16;

% ------------------------------------------------------------
% Create Testbench Stimulus
% ------------------------------------------------------------

Ntransform = 8;
Nfft = N1 * N2 * N3;

TT = numerictype(1,16,0);
FF = fimath('RoundingMethod','Round','OverflowAction','Saturate');

% Generate addressing patterns:
sig_i = zeros(Nfft,Ntransform);
sig_o = zeros(Nfft,Ntransform);
for tt = 1 : Ntransform
  sig_i(:,tt) = compute_addr_3d(N1,N2,N3,2);
  sig_o(:,tt) = compute_addr_3d(N1,N2,N3,3);
end
sig_i = fi(sig_i,TT,FF);
sig_o = fi(sig_o,TT,FF);

% Inputs coming back from DFT-9 with ss0 and ss1 delivering alternating transforms (9 points on each):
data_i = reshape(sig_i,N2,[]);
ss0_i = reshape(data_i(:,1:2:end),1,[]);
ss1_i = reshape(data_i(:,2:2:end),1,[]);

% Outputs send four samples alternately on ss0 and ss1:
data_o = reshape(sig_o,4,[]);
ss0_o = reshape(data_o(:,1:2:end),1,[]);
ss1_o = reshape(data_o(:,2:2:end),1,[]);

% ------------------------------------------------------------
% Write I/O Files
% ------------------------------------------------------------

[~,~,~] = rmdir('data','s');
[~,~,~] = mkdir('data');

fid = fopen('data/ss0_i.txt','w');
for ii = 1 : numel(ss0_i)
  fprintf(fid,'%d %d\n',real(ss0_i(ii)).int,imag(ss0_i(ii)).int);
end
fclose(fid);

fid = fopen('data/ss1_i.txt','w');
for ii = 1 : numel(ss1_i)
  fprintf(fid,'%d %d\n',real(ss1_i(ii)).int,imag(ss1_i(ii)).int);
end
fclose(fid);

fid = fopen('data/ss0_o.txt','w');
for ii = 1 : numel(ss0_o)
  fprintf(fid,'%d %d\n',real(ss0_o(ii)).int,imag(ss0_o(ii)).int);
end
fclose(fid);

fid = fopen('data/ss1_o.txt','w');
for ii = 1 : numel(ss1_o)
  fprintf(fid,'%d %d\n',real(ss1_o(ii)).int,imag(ss1_o(ii)).int);
end
fclose(fid);

