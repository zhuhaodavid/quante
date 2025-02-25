function matlabeig(varargin)
    % Get the number of arguments passed to the function
    nargs = nargin;

    % Get the values of the arguments
    LoadPath = string(varargin(1));
    SavePath = string(varargin(2));
    usegpu = str2double(string(varargin(3)));
    
    fprintf('===============================================================\n');
    t = datetime('now','TimeZone','local','Format','yyyy-MM-dd HH:mm:ss');
    fprintf('time at %s\n', t);
    fprintf('loading from %s\n', LoadPath);
    fprintf('saving to %s\n', SavePath);
    fprintf('\n');
    for i = 4:nargs
        cur_file = string(varargin(i));
        fprintf('dealing with %s\n', cur_file);

        try
            % 尝试加载 .mat 文件
            load(LoadPath + cur_file + '.mat');
        catch
            % 如果加载 .mat 文件失败，尝试加载 HDF5 文件
            fprintf('Failed to load .mat file, trying to load HDF5 file...\n');
            h5_file = LoadPath + cur_file + '.h5';
            try
                info = h5info(h5_file);
                if any(strcmp({info.Datasets.Name}, 'H')) && any(strcmp({info.Datasets.Name}, 'dim'))
                    H = h5read(h5_file, '/H');
                    if isstruct(H) && isfield(H, 'r') && isfield(H, 'i')
                        H = H.r + 1i * H.i;
                        clear H.r H.i; % 释放不再需要的变量
                    end
                    dim = h5read(h5_file, '/dim');
                elseif any(strcmp({info.Datasets.Name}, 'row')) && any(strcmp({info.Datasets.Name}, 'col')) && any(strcmp({info.Datasets.Name}, 'data')) && any(strcmp({info.Datasets.Name}, 'dim'))
                    row = h5read(h5_file, '/row');
                    col = h5read(h5_file, '/col');
                    data = h5read(h5_file, '/data');
                    if isstruct(data) && isfield(data, 'r') && isfield(data, 'i')
                        data = data.r + 1i * data.i;
                        clear data.r data.i; % 释放不再需要的变量
                    end
                    dim = h5read(h5_file, '/dim');
                else
                    error('HDF5 file does not contain the required datasets.');
                end
            catch
                error('Failed to load HDF5 file.');
            end
        end
        dim = double(dim);

        if ~exist('H', 'var')
            fprintf('convert to full ...\n')
            if usegpu
                parallel.gpu.enableCUDAForwardCompatibility(true)
                row = gpuArray(row);
                col = gpuArray(col);
                data = gpuArray(data);
            end
            H0 = sparse(double(row)+1, double(col)+1, double(data), dim, dim);
            Hmatrix123 = full(H0);
        else
            if usegpu
                fprintf('converting to gpu ...\n')
                parallel.gpu.enableCUDAForwardCompatibility(true)
                Hmatrix123 = gpuArray(H);
            else
                Hmatrix123 = H;
            end
        end

        tr0 = trace(Hmatrix123);

        fprintf('eiging ... dim = %d\n', dim);
        tic;
        [psi, E] = eig(Hmatrix123, 'vector');
        psi = psi';
        toc;
        if usegpu
            E = gather(E);
            psi = gather(psi);
        end

        fprintf('trace before: %f,  trace after: %f\n', tr0, sum(E))

        E_path = SavePath + "E_" + cur_file + '.h5';
        if isreal(E)
            h5create(E_path, '/data', [dim], 'Datatype',class(E));
            h5write(E_path, '/data', E);
        else
            h5create(E_path, '/real', [dim], 'Datatype',class(E));
            h5write(E_path, '/real', real(E));
            h5create(E_path, '/imag', [dim], 'Datatype',class(E));
            h5write(E_path, '/imag', imag(E));
        end
        
        psi_path = SavePath + "psi_" + cur_file + '.h5';
        if isreal(psi)
            h5create(psi_path, '/data', [dim,dim], 'Datatype',class(psi));
            h5write(psi_path, '/data', psi);
        else
            h5create(psi_path, '/real', [dim, dim], 'Datatype',class(psi));
            h5write(psi_path, '/real', real(psi));
            h5create(psi_path, '/imag', [dim, dim], 'Datatype',class(psi));
            h5write(psi_path, '/imag', imag(psi));
        end

        fprintf('saved\n')
        fprintf('\n')
        clear psi E
    end
    
    fprintf('finished')

end
