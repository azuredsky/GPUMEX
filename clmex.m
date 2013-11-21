function clmex(clFileName)
%CLMEX Compiles and links a OPENCL file for MATLAB usage
%   CLMEX(FILENAME) will create a MEX-File (also with the name FILENAME) 
%   function in MATLAB.

% Copyright 2009 The MathWorks, Inc.

% !!! Modify the paths below to fit your own installation !!!
if ispc % Windows
    OPENCL_LIB_Location = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\lib\Win32';
    Host_Compiler_Location = '-ccbin "D:\Program Files\Microsoft Visual Studio 11.0\VC\bin"';
    PIC_Option = '';
%   
    aa= 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\include';
    aa1='E:\work\code\NVMEX';
    aa2='D:\Program Files\MATLAB\R2013a\extern\include';
    aa3='D:\boost_1_54_0';
    aa4='D:\Users\sky\GitHub\compute\include';
else % Mac and Linux (assuming gcc is on the path)
    OPENCL_Location = '/usr/local/cuda/lib64';
    Host_Compiler_Location = '';
end
% !!! End of things to modify !!!

[path,filename,zaet] = fileparts(clFileName);
%mex  clFileName -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\include"  -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\lib\Win32"  -lopencl 
  mexCommandLine = [...
     'mex (''' filename '.cpp'' ,''-I' aa  ''',''-I' aa1  ''',''-I' aa2  ''',''-I' aa3  ''',''-I' aa4  ''', ''-L' OPENCL_LIB_Location ''', ''-lopencl'')'];

disp(mexCommandLine);
eval(mexCommandLine);

end
