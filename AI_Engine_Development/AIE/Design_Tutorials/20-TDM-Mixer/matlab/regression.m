%
% Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
% SPDX-License-Identifier: MIT
%
% Author: Mark Rollins

function regression(x86,settings)
   if     (nargin == 0) x86 = 1; settings = [64,32,4];
   elseif (nargin == 1)          settings = [64,32,4];
   elseif (nargin ~= 2) error('Bad voodoo'); end

   close all;
   NSAMP   = settings(1);
   CC      = settings(2);
   NITER   = settings(3);
   if (x86==0)  base = 'aiesimulator_output';
   else         base = 'x86simulator_output';
   end
   % Samples in polyphase order:
   act_o = load_aiesim(sprintf('../aie/%s/data/sig_o.txt',base),'int',1);
   gld_o = load_aiesim(        '../aie/data/sig_o.txt',         'int',1);

   act_o = reshape(act_o,CC,NSAMP,NITER);
   gld_o = reshape(gld_o,CC,NSAMP,NITER);
   
   err = reshape(act_o - gld_o,1,[]);
   
   if (1)
     h = figure;
     set(h,'Position',[273,153,1493,686]);
     rr = 2^floor(log2(floor(sqrt(CC))));
     cc = CC/rr;
     ii = 1;
% $$$      plot(reshape(real(gld_o(ii,:,:)),1,[]),'b.-' ); hold on;
% $$$      plot(reshape(real(act_o(ii,:,:)),1,[]),'r.--'); hold off;
     for yy = 1 : rr
       for xx = 1 : cc
         subplot(rr,cc,ii); plot(reshape(real(gld_o(ii,:,:)),1,[]),'b.-' ); hold on;
         subplot(rr,cc,ii); plot(reshape(real(act_o(ii,:,:)),1,[]),'r.--'); hold off;
         ii = ii + 1;
       end
     end
   end
   max_err = max([abs(real(err)),abs(imag(err))]);
   fprintf(1,'Max Err: %g\n',max_err);
   level = 5;
   if ( max_err <= level )
     fprintf(1,'--- PASSED ---\n');
   else
     fprintf(1,'*** FAILED ***\n');
   end
end
