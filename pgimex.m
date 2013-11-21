function nvmex(cuFileName)
%NVMEX Compiles and links a CUDA file for MATLAB usage
%   NVMEX(FILENAME) will create a MEX-File (also with the name FILENAME) by
%   invoking the CUDA compiler, nvcc, and then linking with the MEX
%   function in MATLAB.

% Copyright 2009 The MathWorks, Inc.

% !!! Modify the paths below to fit your own installation !!!
if ispc % Windows
    CUDA_LIB_Location = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\lib\Win32';
    PGI_ACC_LIB='D:\Program Files\PGI\win32\13.9\lib';
    Host_Compiler_Location = '"D:\Program Files\PGI\win32\13.9\bin"';
    PIC_Option = '';
    aa=' -I"D:\Program Files\MATLAB\R2013a\extern\include","E:\work\code\NVMEX"';
else % Mac and Linux (assuming gcc is on the path)
    CUDA_LIB_Location = '/usr/local/cuda/lib64';
    Host_Compiler_Location = '';
    PIC_Option = ' --compiler-options -fPIC ';
end
% !!! End of things to modify !!!

[path,filename,zaet] = fileparts(cuFileName);

nvccCommandLine = [ ...
    'pgcc -acc -tp=nvidia -Minfo=accel -fast -c -Bdynamic ' cuFileName  ...
    ' -o ' filename '.o ' ...
    PIC_Option...
    aa
    ];

mexCommandLine = ['mex (''' filename '.o'',  ''-L' CUDA_LIB_Location ''', ''-L' PGI_ACC_LIB ''', ''-lcudart'',''-lcufft'', ''-lcublas'',''-llibaccapi'',''-llibacc*'',''-lstd'',''-llibacc1'')'];

disp(nvccCommandLine);
status = system(nvccCommandLine);
if status < 0
    error 'Error invoking nvcc';
end

disp(mexCommandLine);
eval(mexCommandLine);

end
