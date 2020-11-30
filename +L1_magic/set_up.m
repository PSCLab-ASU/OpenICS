% Run this script prior to using the framework
% Sets up the framework environment

[folder, filename, extension] = fileparts(mfilename('fullpath'));

% Add l1magic folders to path
path(path, strcat(folder, '/l1magic/Measurements'));
path(path, strcat(folder, '/l1magic/Optimization'));
disp('Finished L1-magic set up!');
