% Run this script prior to using the framework
% Sets up the framework environment

[folder, filename, extension] = fileparts(mfilename('fullpath'));
original_folder = pwd;

% Add TVAL3 folders to the path
path(path, strcat(folder, '/TVAL3_v1.0/Fast_Walsh_Hadamard_Transform'));
path(path, strcat(folder, '/TVAL3_v1.0/Solver'));
path(path, strcat(folder, '/TVAL3_v1.0/Utilities'));

% Compile TVAL3's fWHT function
cd(folder);
cd TVAL3_v1.0/Fast_Walsh_Hadamard_Transform;
mex -O fWHtrans.cpp
fprintf('Finished compiling the C++ code for fast Walsh-Hadamard transform!\n');
cd(original_folder);
