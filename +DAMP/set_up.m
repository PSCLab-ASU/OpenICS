% Run this script prior to using the framework
% Sets up the framework environment

[folder, filename, extension] = fileparts(mfilename('fullpath'));
original_folder = pwd;

% Add D-AMP folders to the path
path(path, strcat(folder, '/D-AMP_Toolbox/Algorithms'));
path(path, genpath(strcat(folder, '/D-AMP_Toolbox/Packages')));

% Compile D-AMP's rwt toolbox
cd(folder);
cd D-AMP_Toolbox/Packages/rwt/bin;
compile;
cd(original_folder);
fprintf('Finished compiling the C++ code for the rwt toolbox!\n');
