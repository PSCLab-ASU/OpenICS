function [] = create_path()
    % obtain current file's folder
    [folder, filename, extension] = fileparts(mfilename('fullpath'));
    
    % add l1-magic folders to path
    path(path, strcat(folder, '/l1magic/Measurements'));
    path(path, strcat(folder, '/l1magic/Optimization'));
end
