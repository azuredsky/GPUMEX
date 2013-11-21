clear all;
close all;
clc;
format compact;
format long e;

N = 2^17

ns=300

% m/L must be integer
% m >= L
% ns > m
L = 1
m = 600 % 600

if mod(m,L) ~= 0,
    error('m/L must be integer')
end

if m < L,
    error('must be: m > L')
end


% load('x1.mat')
% load('x2.mat')
load( 'cafdata.mat')

x1 = vPriK(1:N);
x2 = vOdrK(1:N);
x22 = zeros(1,N+m);
x22(1:N) = x2(1:N);

x2 = conj(x2);
x22 = conj(x22);

sx = single( zeros(1,N));
sxfref = single( zeros(m,2*ns) );


disp('*** Matlab routine ***')
tic;
for tau = 0:(m-1), % 1-index
    sx =  single(x1 .* x22(1+tau:N+tau)) ;
    sxfft = single(fft(sx));
    sxfref(tau+1, 1:ns) = sxfft(1:ns);
    sxfref(tau+1, ns+1:2*ns) = sxfft(N-ns+1:N);
end
toc;

disp('*** CUDA routine ***')
tic;
sxf = caf_cuda(x1, x2, m, L, ns);
toc;

minsxerr = zeros(m,1);
maxsxerr = zeros(m,1);

for ii = 1:m
    tmperr = abs(sxf(ii,:) - sxfref(ii,:))./abs(sxfref(ii,:));
    minsxerr(ii) = min(tmperr);
    maxsxerr(ii) = max(tmperr);
end


if 0
    for tau=0:(m-1)
        figure(tau+1);
        subplot(211);stem(abs(sxfref(tau+1,1:2*ns)),'.');
        title(['\tau = ', num2str(tau)]);
        subplot(212);stem(abs(sxf(tau+1,1:2*ns)),'.');
        title(['\tau = ', num2str(tau)]);
    end
end

figure;
subplot(211);stem(0:1:m-1, maxsxerr,'.');
title('Maximal Normalized Error');
subplot(212);stem(0:1:m-1, minsxerr,'.');
title('Minimal Normalized Error');


disp('*** Error ***')
if m<21
    [maxsxerr minsxerr]
else
    [max(maxsxerr) max(minsxerr)]
end

% x-axis
fval = ( -ns+1:1:ns );
% y-axis
tauval = 0:m-1;

pisvejcConst = -65  % dB
sxfa = double( 10.*log10( (sxf.*conj(sxf))./N) ) + pisvejcConst;
% swap negative freq.
sxfag = zeros(m,2*ns);
sxfag(:,1:ns)      = sxfa(:,ns+1:2*ns); % Fs-ns up to Fs-1
sxfag(:,ns+1:2*ns) = sxfa(:,1:ns);      % 0 up to ns

pisvejcClipping = -180;
nefungujeMiClipping = sxfag .* (sxfag > pisvejcClipping) + (pisvejcClipping) .* (sxfag <= pisvejcClipping);
cmin = min(min(sxfag)) - pisvejcConst/3*2 +5
cmax = max(max(sxfag)) + pisvejcConst/4 - 5

figure;
% otevreni figury pro kresleni a jeji umisteni
set(gcf,'Position',[315 241 783 693]);
% volba barevne skaly
% colormap('default'); cm = colormap;
% ww = size( cm); cm = cm(1:ww(1)-5, :);
% colormap( cm);

% kresleni
h = surf(fval, tauval, nefungujeMiClipping); %it's faster
set(h,'EdgeColor','interp');
set(h,'FaceColor','interp');
set(h,'FaceLight','phong');

set(gca,'view',[-20 25])
set(gca,'Clim',[cmin cmax]);

% surf(fval, tauval, sxfag, 'EdgeColor','interp'); %it's faster
xlabel('f [Hz]'); ylabel('tau [-]'); zlabel('CAF [dB]')
set(gca,'XDir', 'norm','YDir','rev')
% set(gca,'ZLim',([ ]))
set(gca, 'Clipping', 'on')



