% Run this script prior to using the framework
% Sets up the framework environment

[folder, filename, extension] = fileparts(mfilename('fullpath'));

% Add D-AMP folders to the path
path(path, strcat(folder, '/D-AMP_Toolbox/Algorithms'));
path(path, genpath(strcat(folder, '/D-AMP_Toolbox/Packages')));

% Compile TVAL3's fWHT function
cd D-AMP_Toolbox/Packages/rwt/bin;
compile;
cd ..;
cd ..;
cd ..;
cd ..;
fprintf('Finished compiling the C++ code for the rwt toolbox!\n');
