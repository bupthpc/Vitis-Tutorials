%
% Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
% SPDX-License-Identifier: MIT
%
% Author: Faisal El-Shabani

clear all;
close all;

% ------------------------------------------------------------
% Create Permutation Mapping
% ------------------------------------------------------------

NSTREAM = 8;
NFFT_1D = 64;
NFFT = NFFT_1D * NFFT_1D;
NSAMP = NFFT_1D;

% ------------------------------------------------------------
% Create Testbench Stimulus
% ------------------------------------------------------------

% Generate walking sequence pattern for easy testing:
seq_i = [0:NFFT-1];                     % These samples are a proxy for 'cint32' values (ie. 64-bits each)
tmp = reshape(seq_i,NFFT_1D,NFFT_1D);
sig_i = tmp;

% ------------------------------------------------------------
% Write I/O Files
% ------------------------------------------------------------

[~,~,~] = rmdir('data','s');
[~,~,~] = mkdir('data');

% Write streams in polyphase order:
for ss = 1 : NSTREAM
  fid = fopen(sprintf('data/stream%d_i.txt',ss-1),'w');
  data = reshape(sig_i,NSTREAM,[]);
  for rr = 1 : size(data(ss,:),2)
    fprintf(fid,'%d\n',data(ss,rr));
  end
  fclose(fid);
end

% Write streams taken row-wise in polyphase order:
for ss = 1 : NSTREAM
  data = sig_i(ss:NSTREAM:end,:);
  fid = fopen(sprintf('data/stream%d_o.txt',ss-1),'w');
  for ii = 1 : size(data,1)
    for jj = 1 : size(data,2)
      fprintf(fid,'%d\n',data(ii,jj));
    end
  end
    fclose(fid);
end
