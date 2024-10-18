
function matlabeigvals(varargin)
    % Get the number of arguments passed to the function
    nargs = nargin;

    % Get the values of the arguments
    LoadPath = string(varargin(1));
    SavePath = string(varargin(2));
    
    fprintf('===============================================================\n');
    t = datetime('now','TimeZone','local','Format','yyyy-MM-dd HH:mm:ss');
    fprintf('time at %s\n', t);
    fprintf('loading from %s\n', LoadPath);
    fprintf('saving to %s\n', SavePath);
    fprintf('\n');
    for i = 3:nargs
        cur_file = string(varargin(i));
        fprintf('dealing with %s\n', cur_file);

        load(LoadPath + cur_file + '.mat');
        dim = double(dim);

        if ~exist('H', 'var')
            fprintf('convert to full ...\n')
            H0 = sparse(double(row)+1, double(col)+1, double(data), dim, dim);
            Hmatrix123 = full(H0);
        else
            Hmatrix123 = H;
        end
        tr0 = trace(Hmatrix123);

        fprintf('eigvalsing ... dim = %d\n', dim);
        tic;
        E = eig(Hmatrix123);
        toc;

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

        fprintf('saved\n')
        fprintf('\n')
    end
    
    fprintf('finished')

end

