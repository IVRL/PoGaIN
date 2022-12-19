% PoGaIN: Poisson-Gaussian Image Noise Modeling from Paired Samples

% Authors: Nicolas Bähler, Majed El Helou, Étienne Objois, Kaan Okumuş, and Sabine
% Süsstrunk, Fellow, IEEE.

function get_preds(dir_name)
    addpath(strcat('code/denoise/matlab/utils/'));

    outputFolder = fullfile(cd, strcat('results/', dir_name));

    if exist(outputFolder, 'dir')
        rmdir(outputFolder, 's');
    end

    mkdir(outputFolder);

    D = convertStringsToChars(strcat(cd, '/data/', dir_name));

    S = dir(fullfile(D, '*.mat'));

    NoiseAB = zeros(numel(S), 2);
    FileNames = strings(numel(S), 1);

    for img_idx = 1:numel(S)
        img_name = S(img_idx).name;
        FileNames(img_idx) = convertCharsToStrings(img_name);

        F = fullfile(D, img_name);
        y = load(F);
        y = y.data;

        % Estimating the noise
        fitparams = estimate_noise(y);
        a = fitparams(1);
        b = fitparams(2);

        % if a < 0 %TODO Why needed?
        %     a = eps;
        % end

        % if b < 0
        %     b = eps;
        % end

        if a == 0 % TODO What happens if a < 0?
            a = Inf;
        else
            a = 1 / a;
        end

        if b <= 0
            b = 0;
        else
            b = sqrt(b);
        end

        NoiseAB(img_idx, 1) = a;
        NoiseAB(img_idx, 2) = b;

    end

    save_name = strcat('results/', dir_name, '/NoiseAB.mat');
    save(save_name, 'NoiseAB');
    save_name = strcat('results/', dir_name, '/FileNames.txt');
    fid = fopen(save_name, 'wt');
    fprintf(fid, '%s\n', FileNames{:});
    fclose(fid);
end
