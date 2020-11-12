% Run this script prior to using the framework
% Sets up the framework environment

[folder, filename, extension] = fileparts(mfilename('fullpath'));

% Add NLR_CS folders to path
path(path, strcat(folder, '/NLR_codes/NLR_CS/Utilities'));
path(path, strcat(folder, '/NLR_codes/NLR_CS/Utilities/Measurements'));
