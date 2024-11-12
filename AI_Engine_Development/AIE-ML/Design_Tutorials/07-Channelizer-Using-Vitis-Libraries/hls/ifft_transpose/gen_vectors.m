%
% Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
% SPDX-License-Identifier: MIT
%
% Author: Mark Rollins

clear all;
close all;

% ------------------------------------------------------------
% Create Permutation Mapping
% ------------------------------------------------------------

Ntransform = 1;
NSTREAM = 8;
NFFT_1D = 64;
NFFT = NFFT_1D * NFFT_1D;
NSAMP = NFFT_1D;

% ------------------------------------------------------------
% Create Testbench Stimulus
% ------------------------------------------------------------

pattern = 1;

seq_i = zeros(NFFT_1D,NFFT_1D,Ntransform);

if (pattern == 1 )
  % Use pattern to prove out functionality:
  for tt = 1 : Ntransform
    tmp = reshape(compute_addr_2d(NFFT_1D,NFFT_1D,1),NFFT_1D,NFFT_1D);
    seq_i(:,:,tt) = tmp;
  end
else
  limit = 2^16-1;
  seq_i = randi([-(limit+1),limit],NSAMP,NSAMP,Ntransform);
end
  
% ------------------------------------------------------------
% Write I/O Files
% ------------------------------------------------------------

[~,~,~] = rmdir('data','s');
[~,~,~] = mkdir('data');


% Write streams in polyphase order:
for ss = 1 : NSTREAM
  fid = fopen(sprintf('data/stream%d_i.txt',ss-1),'a');
  for tt = 1 : Ntransform
    data = seq_i(ss:NSTREAM:end,:,tt);
    for rr = 1 : size(data,1)
      for cc = 1 : size(data,2)
        % Data is coming in along rows:
        fprintf(fid,'%d\n',data(rr,cc));
      end
    end
  end
  fclose(fid);
end

% Write streams in polyphase order:
for ss = 1 : NSTREAM
  fid = fopen(sprintf('data/stream%d_o.txt',ss-1),'a');
  for tt = 1 : Ntransform
    data = seq_i(:,ss:NSTREAM:end,tt);
    for cc = 1 : size(data,2)
      for rr = 1 : size(data,1)
        % Data is leaving along columns:
        fprintf(fid,'%d\n',data(rr,cc));
      end
    end
  end
  fclose(fid);
end