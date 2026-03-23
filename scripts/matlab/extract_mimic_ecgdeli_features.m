function extract_mimic_ecgdeli_features(index_csv, output_csv, num_workers, start_idx, end_idx, append_output)
% ECGDeli Feature Extraction for MIMIC-IV
% Extracts 93 validated ECGDeli features and writes CSV keyed by study_id.

ecgdeli_path = getenv('ECGDELI_PATH');
wfdb_path = getenv('WFDB_PATH');
mimic_root = getenv('MIMIC_ECG_ROOT');

if ~isempty(ecgdeli_path) && isfolder(ecgdeli_path)
    addpath(genpath(ecgdeli_path));
    fprintf('Added ECGDeli path: %s\n', ecgdeli_path);
end
if ~isempty(wfdb_path) && isfolder(wfdb_path)
    addpath(genpath(wfdb_path));
    fprintf('Added WFDB path: %s\n', wfdb_path);
end

if ~isempty(mimic_root) && isfolder(mimic_root)
    cd(mimic_root);
    fprintf('Using MIMIC root as MATLAB cwd: %s\n', mimic_root);
end

mimic_root_resolved = string(mimic_root);
if strlength(mimic_root_resolved) == 0 || ~isfolder(char(mimic_root_resolved))
    mimic_root_resolved = string(pwd);
end

if exist('setwfdb', 'file') == 2 && strlength(mimic_root_resolved) > 0 && isfolder(char(mimic_root_resolved))
    try
        setwfdb(char(mimic_root_resolved));
        fprintf('Configured WFDB search root: %s\n', mimic_root_resolved);
    catch wfdbSetME
        fprintf('Warning: failed to configure WFDB root (%s)\n', wfdbSetME.message);
    end
end

required_functions = {
    'rdsamp', 'Annotate_ECG_Multi', 'ExtractAmplitudeFeaturesFromFPT', ...
    'ExtractIntervalFeaturesFromFPT', 'ECG_High_Low_Filter', ...
    'Notch_Filter', 'Isoline_Correction', 'ECG_Baseline_Removal'
};
for f = 1:numel(required_functions)
    if exist(required_functions{f}, 'file') ~= 2
        error('Required MATLAB function not found on path: %s', required_functions{f});
    end
end

fprintf('==========================================================\n');
fprintf('ECGDeli 1.1 Feature Extraction for MIMIC-IV\n');
fprintf('==========================================================\n\n');

fprintf('Loading index: %s\n', index_csv);
cohort = readtable(index_csv);

if nargin < 4 || isempty(start_idx)
    start_idx = 1;
end
if nargin < 5 || isempty(end_idx)
    end_idx = 0;
end
if nargin < 6 || isempty(append_output)
    append_output = false;
end

start_idx = max(1, floor(start_idx));
end_idx = floor(end_idx);
append_output = logical(append_output);

required_cols = {'study_id'};
for i = 1:numel(required_cols)
    if ~ismember(required_cols{i}, cohort.Properties.VariableNames)
        error('Missing required column in index CSV: %s', required_cols{i});
    end
end

if ~ismember('waveform_path', cohort.Properties.VariableNames)
    error('index CSV must contain waveform_path column exported by Python pipeline.');
end

total_records = height(cohort);
if end_idx <= 0
    end_idx = total_records;
end
end_idx = min(end_idx, total_records);

if start_idx > total_records || start_idx > end_idx
    error('Requested ECGDeli range is empty/invalid (start=%d, end=%d, total=%d).', start_idx, end_idx, total_records);
end

cohort = cohort(start_idx:end_idx, :);
num_records = height(cohort);
fprintf('  Loaded %d ECG records (total=%d, range=%d..%d)\n\n', num_records, total_records, start_idx, end_idx);

if nargin < 3 || isempty(num_workers)
    workers_env = str2double(getenv('MATLAB_ECGDELI_WORKERS'));
    if isnan(workers_env) || workers_env < 1
        num_workers = 1;
    else
        num_workers = floor(workers_env);
    end
end
num_workers = max(1, floor(num_workers));

feature_names = {
    'R_Amp_I', 'R_Amp_I_count', 'R_Amp_I_iqr', ...
    'R_Amp_II', 'R_Amp_II_count', 'R_Amp_II_iqr', ...
    'R_Amp_III', 'R_Amp_III_count', 'R_Amp_III_iqr', ...
    'R_Amp_V3', 'R_Amp_V3_count', 'R_Amp_V3_iqr', ...
    'R_Amp_V4', 'R_Amp_V4_count', 'R_Amp_V4_iqr', ...
    'R_Amp_V5', 'R_Amp_V5_count', 'R_Amp_V5_iqr', ...
    'R_Amp_V6', 'R_Amp_V6_count', 'R_Amp_V6_iqr', ...
    'R_Amp_aVF', 'R_Amp_aVF_count', 'R_Amp_aVF_iqr', ...
    'R_Amp_aVL', 'R_Amp_aVL_count', 'R_Amp_aVL_iqr', ...
    'S_Amp_V5', 'S_Amp_V5_count', 'S_Amp_V5_iqr', ...
    'S_Amp_V6', 'S_Amp_V6_count', 'S_Amp_V6_iqr', ...
    'T_Amp_I', 'T_Amp_I_count', 'T_Amp_I_iqr', ...
    'T_Amp_II', 'T_Amp_II_count', 'T_Amp_II_iqr', ...
    'T_Amp_III', 'T_Amp_III_count', 'T_Amp_III_iqr', ...
    'T_Amp_aVR', 'T_Amp_aVR_count', 'T_Amp_aVR_iqr', ...
    'T_Amp_aVL', 'T_Amp_aVL_count', 'T_Amp_aVL_iqr', ...
    'T_Amp_aVF', 'T_Amp_aVF_count', 'T_Amp_aVF_iqr', ...
    'T_Amp_V1', 'T_Amp_V1_count', 'T_Amp_V1_iqr', ...
    'T_Amp_V2', 'T_Amp_V2_count', 'T_Amp_V2_iqr', ...
    'T_Amp_V3', 'T_Amp_V3_count', 'T_Amp_V3_iqr', ...
    'T_Amp_V4', 'T_Amp_V4_count', 'T_Amp_V4_iqr', ...
    'T_Amp_V5', 'T_Amp_V5_count', 'T_Amp_V5_iqr', ...
    'T_Amp_V6', 'T_Amp_V6_count', 'T_Amp_V6_iqr', ...
    'ST_Elev_II', 'ST_Elev_II_count', 'ST_Elev_II_iqr', ...
    'ST_Elev_III', 'ST_Elev_III_count', 'ST_Elev_III_iqr', ...
    'ST_Elev_aVR', 'ST_Elev_aVR_count', 'ST_Elev_aVR_iqr', ...
    'ST_Elev_aVF', 'ST_Elev_aVF_count', 'ST_Elev_aVF_iqr', ...
    'RR_Mean_Global', 'RR_Mean_Global_count', 'RR_Mean_Global_iqr', ...
    'QRS_Dur_Global', 'QRS_Dur_Global_count', 'QRS_Dur_Global_iqr', ...
    'QT_IntFramingham_Global', 'QT_IntFramingham_Global_count', 'QT_IntFramingham_Global_iqr', ...
    'PR_Int_Global', 'PR_Int_Global_count', 'PR_Int_Global_iqr'
};

results = table();
results.study_id = cohort.study_id;

feature_count = length(feature_names);
feature_matrix = nan(num_records, feature_count);
success_flags = false(num_records, 1);
error_messages = strings(num_records, 1);

checkpoint_every = 500;
checkpoint_env = str2double(getenv('MATLAB_ECGDELI_CHECKPOINT_EVERY'));
if ~isnan(checkpoint_env) && checkpoint_env > 0
    checkpoint_every = floor(checkpoint_env);
end
output_folder = fileparts(output_csv);
if ~isempty(output_folder) && ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

fprintf('Initialized results table with %d features\n\n', length(feature_names));
fprintf('Starting ECGDeli extraction...\n');
fprintf('----------------------------------------------------------\n');
fprintf('Workers requested: %d\n', num_workers);
fprintf('Checkpoint every: %d records\n', checkpoint_every);
fprintf('Append output mode: %d\n', append_output);

baseline_results = table();
if append_output
    output_rows = 0;
    output_results = table();
    if isfile(output_csv)
        output_results = readtable(output_csv);
        output_rows = height(output_results);
    end

    checkpoint_pattern = fullfile(output_folder, 'checkpoint_ecgdeli_*.csv');
    checkpoint_listing = dir(checkpoint_pattern);
    best_checkpoint_rows = 0;
    best_checkpoint_file = "";
    best_checkpoint_results = table();

    for ci = 1:numel(checkpoint_listing)
        token = regexp(checkpoint_listing(ci).name, '^checkpoint_ecgdeli_(\d+)\.csv$', 'tokens', 'once');
        if isempty(token)
            continue;
        end
        checkpoint_rows = str2double(token{1});
        if isnan(checkpoint_rows) || checkpoint_rows >= start_idx || checkpoint_rows <= best_checkpoint_rows
            continue;
        end

        candidate_file = string(fullfile(checkpoint_listing(ci).folder, checkpoint_listing(ci).name));
        if ~isfile(char(candidate_file))
            continue;
        end

        candidate_results = readtable(char(candidate_file));
        candidate_row_count = height(candidate_results);
        if candidate_row_count < checkpoint_rows
            fprintf('Ignoring malformed checkpoint for baseline: %s (rows=%d, expected>=%d)\n', candidate_file, candidate_row_count, checkpoint_rows);
            continue;
        end

        best_checkpoint_rows = checkpoint_rows;
        best_checkpoint_file = candidate_file;
        best_checkpoint_results = candidate_results;
    end

    if best_checkpoint_rows > output_rows
        baseline_results = best_checkpoint_results;
        fprintf('Loaded baseline checkpoint for resume checkpoints: %s (%d rows)\n', best_checkpoint_file, height(baseline_results));
    elseif output_rows > 0
        baseline_results = output_results;
        fprintf('Loaded baseline output for resume checkpoints: %d rows\n', height(baseline_results));
    elseif best_checkpoint_rows > 0
        baseline_results = best_checkpoint_results;
        fprintf('Loaded baseline checkpoint for resume checkpoints: %s (%d rows)\n', best_checkpoint_file, height(baseline_results));
    end
end

tic;
success_count = 0;
fail_count = 0;

can_parallel = false;
if num_workers > 1
    has_parallel_toolbox = ~isempty(ver('parallel')) && license('test', 'Distrib_Computing_Toolbox');
    if has_parallel_toolbox
        try
            local_cluster = parcluster('local');
            profile_max_workers = local_cluster.NumWorkers;
            target_workers = num_workers;

            if target_workers > profile_max_workers
                try
                    local_cluster.NumWorkers = target_workers;
                    fprintf('Runtime local worker cap raised: %d -> %d\n', profile_max_workers, local_cluster.NumWorkers);
                catch workerCapME
                    fprintf('Could not raise runtime local worker cap (%s). Using max available %d workers.\n', workerCapME.message, profile_max_workers);
                    target_workers = profile_max_workers;
                end
            end

            target_workers = max(1, min(target_workers, local_cluster.NumWorkers));
            pool = gcp('nocreate');
            if isempty(pool)
                parpool(local_cluster, target_workers);
            elseif pool.NumWorkers ~= target_workers
                delete(pool);
                parpool(local_cluster, target_workers);
            end
            can_parallel = true;
            fprintf('Parallel mode: enabled (%d workers requested, %d workers active)\n', num_workers, target_workers);
        catch ME
            fprintf('Parallel mode unavailable (%s). Falling back to sequential mode.\n', ME.message);
            can_parallel = false;
        end
    else
        fprintf('Parallel toolbox/license not available. Falling back to sequential mode.\n');
    end
end

if can_parallel
    block_size = max(1, checkpoint_every);
    for block_start = 1:block_size:num_records
        block_end = min(num_records, block_start + block_size - 1);
        block_indices = block_start:block_end;
        block_n = numel(block_indices);

        block_features = nan(block_n, feature_count);
        block_success = false(block_n, 1);
        block_errors = strings(block_n, 1);

        parfor bi = 1:block_n
            idx = block_indices(bi);
            study_id = cohort.study_id(idx);
            record_path_raw = cohort.waveform_path(idx);
            [row_values, ok, err_msg] = extract_single_ecgdeli_row(record_path_raw, study_id, feature_names, mimic_root_resolved);
            block_features(bi, :) = row_values;
            block_success(bi) = ok;
            block_errors(bi) = string(err_msg);
        end

        feature_matrix(block_indices, :) = block_features;
        success_flags(block_indices) = block_success;
        error_messages(block_indices) = block_errors;

        success_count = sum(success_flags(1:block_end));
        fail_count = block_end - success_count;

        elapsed = toc;
        rate = block_end / max(elapsed, 1e-6);
        eta_seconds = (num_records - block_end) / max(rate, 1e-6);
        eta_hours = eta_seconds / 3600;
        global_block_end = start_idx + block_end - 1;
        fprintf('[%d/%d] %.1f ECGs/min | Success: %d | Failed: %d | ETA: %.1fh\n', ...
            global_block_end, total_records, rate*60, success_count, fail_count, eta_hours);

        block_failed_mask = ~block_success;
        if any(block_failed_mask)
            failed_errors = block_errors(block_failed_mask);
            non_empty_errors = failed_errors(strlength(failed_errors) > 0);
            max_error_samples = min(3, numel(non_empty_errors));
            for ei = 1:max_error_samples
                fprintf('  Sample error: %s\n', non_empty_errors(ei));
            end
        end

        if success_count > 0 && mod(global_block_end, checkpoint_every) == 0
            checkpoint_results = table();
            checkpoint_results.study_id = results.study_id(1:block_end);
            for fi = 1:feature_count
                checkpoint_results.(feature_names{fi}) = feature_matrix(1:block_end, fi);
            end
            if append_output && ~isempty(baseline_results)
                merged_checkpoint = [baseline_results; checkpoint_results];
                [~, keep_idx] = unique(merged_checkpoint.study_id, 'last');
                checkpoint_results = merged_checkpoint(sort(keep_idx), :);
            end
            checkpoint_file = fullfile(output_folder, sprintf('checkpoint_ecgdeli_%d.csv', global_block_end));
            writetable(checkpoint_results, checkpoint_file);
            fprintf('  ✓ Checkpoint saved: %s\n', checkpoint_file);
        end
    end
else
    for idx = 1:num_records
        if mod(idx, 50) == 0 || idx == 1
            elapsed = toc;
            rate = idx / max(elapsed, 1e-6);
            eta_seconds = (num_records - idx) / max(rate, 1e-6);
            eta_hours = eta_seconds / 3600;
            global_idx = start_idx + idx - 1;
            fprintf('[%d/%d] %.1f ECGs/min | Success: %d | Failed: %d | ETA: %.1fh\n', ...
                global_idx, total_records, rate*60, success_count, fail_count, eta_hours);
        end

        study_id = cohort.study_id(idx);
        record_path_raw = cohort.waveform_path(idx);
        [row_values, ok, err_msg] = extract_single_ecgdeli_row(record_path_raw, study_id, feature_names, mimic_root_resolved);

        feature_matrix(idx, :) = row_values;
        success_flags(idx) = ok;
        error_messages(idx) = string(err_msg);

        if ok
            success_count = success_count + 1;
        else
            if fail_count < 5 || mod(fail_count, 10) == 0
                fprintf('  Error (study %d): %s\n', study_id, err_msg);
            end
            fail_count = fail_count + 1;
        end

        global_idx = start_idx + idx - 1;
        if success_count > 0 && mod(global_idx, checkpoint_every) == 0
            checkpoint_results = table();
            checkpoint_results.study_id = results.study_id(1:idx);
            for fi = 1:feature_count
                checkpoint_results.(feature_names{fi}) = feature_matrix(1:idx, fi);
            end
            if append_output && ~isempty(baseline_results)
                merged_checkpoint = [baseline_results; checkpoint_results];
                [~, keep_idx] = unique(merged_checkpoint.study_id, 'last');
                checkpoint_results = merged_checkpoint(sort(keep_idx), :);
            end
            checkpoint_file = fullfile(output_folder, sprintf('checkpoint_ecgdeli_%d.csv', global_idx));
            writetable(checkpoint_results, checkpoint_file);
            fprintf('  ✓ Checkpoint saved: %s\n', checkpoint_file);
        end
    end
end

for fi = 1:feature_count
    results.(feature_names{fi}) = feature_matrix(:, fi);
end

failed_indices = find(~success_flags);
for k = 1:min(numel(failed_indices), 5)
    fidx = failed_indices(k);
    if strlength(error_messages(fidx)) > 0
        fprintf('  Sample error (study %d): %s\n', results.study_id(fidx), error_messages(fidx));
    end
end

fprintf('\n----------------------------------------------------------\n');
fprintf('Saving final results...\n');
if success_count == 0
    error('MATLAB ECGDeli extraction produced 0 successful records. Aborting without writing output to avoid false resume progress.');
end
if append_output && isfile(output_csv)
    existing_results = readtable(output_csv);
    merged_results = [existing_results; results];
    [~, keep_idx] = unique(merged_results.study_id, 'last');
    merged_results = merged_results(sort(keep_idx), :);
    writetable(merged_results, output_csv);
    fprintf('Appended range into existing output (%d total rows after merge)\n', height(merged_results));
else
    writetable(results, output_csv);
end

elapsed_total = toc;
fprintf('\n==========================================================\n');
fprintf('EXTRACTION COMPLETE\n');
fprintf('==========================================================\n');
fprintf('Total ECGs (this run): %d\n', height(cohort));
fprintf('Successful: %d (%.1f%%)\n', success_count, 100*success_count/max(height(cohort), 1));
fprintf('Failed: %d (%.1f%%)\n', fail_count, 100*fail_count/max(height(cohort), 1));
fprintf('Total time: %.1f hours\n', elapsed_total/3600);
fprintf('Average rate: %.1f ECGs/minute\n', height(cohort)/max(elapsed_total/60, 1e-6));
fprintf('Output file: %s\n', output_csv);
fprintf('==========================================================\n');
end


function [row_values, is_success, error_message] = extract_single_ecgdeli_row(record_path_raw, study_id, feature_names, mimic_root)
row_values = nan(1, length(feature_names));
is_success = false;
error_message = "";

if nargin < 4 || strlength(string(mimic_root)) == 0
    mimic_root = string(getenv('MIMIC_ECG_ROOT'));
end

try
    if strlength(mimic_root) > 0 && isfolder(char(mimic_root))
        cd(char(mimic_root));
    end

    record_path = string(record_path_raw);
    if strlength(record_path) == 0
        error('Empty waveform_path');
    end

    record_path = erase(record_path, ".hea");
    record_path = erase(record_path, ".dat");
    record_path = replace(record_path, "\\", "/");

    if ~contains(record_path, ":") && ~startsWith(record_path, "files/")
        record_path = "files/" + erase(record_path, "./");
    end

    [resolved_record_path, path_ok, path_msg] = resolve_local_record_path(record_path, mimic_root);
    if ~path_ok
        error(path_msg);
    end

    [signal, Fs, ~] = rdsamp(char(resolved_record_path));

    if size(signal, 2) ~= 12
        error('Expected 12 leads, got %d', size(signal, 2));
    end

    [~, ~] = ECG_Baseline_Removal(signal, Fs, 1, 0.5);
    ecg_filtered_frq = ECG_High_Low_Filter(signal, Fs, 1, 40);
    ecg_filtered_frq = Notch_Filter(ecg_filtered_frq, Fs, 50, 1);
    [ecg_filtered, ~, ~, ~] = Isoline_Correction(ecg_filtered_frq);

    [FPT_MultiChannel, FPT_Cell] = Annotate_ECG_Multi(ecg_filtered, Fs);
    Amplitude_features = ExtractAmplitudeFeaturesFromFPT(FPT_Cell, ecg_filtered);
    [Timing_leadwise, Timing_sync] = ExtractIntervalFeaturesFromFPT(FPT_Cell, FPT_MultiChannel);

    features_struct = extract_93_features(Amplitude_features, Timing_leadwise, Timing_sync, FPT_Cell, ecg_filtered);
    fn = fieldnames(features_struct);
    for j = 1:numel(fn)
        feat_name = fn{j};
        feat_idx = find(strcmp(feature_names, feat_name), 1);
        if ~isempty(feat_idx)
            row_values(feat_idx) = features_struct.(feat_name);
        end
    end

    is_success = true;
catch ME
    error_message = string(ME.message);
    if ~isempty(study_id)
        error_message = "study " + string(study_id) + ": " + error_message;
    end
end


function [resolved_path, ok, message] = resolve_local_record_path(record_path, mimic_root)
resolved_path = string(record_path);
ok = false;
message = "";

if strlength(resolved_path) == 0
    message = "Empty waveform_path after normalization";
    return;
end

resolved_path = strip(resolved_path);
resolved_path = replace(resolved_path, "\\", "/");
resolved_path = regexprep(resolved_path, '/+$', '');

if strlength(resolved_path) == 0
    message = "Empty waveform_path after trimming";
    return;
end

if ~contains(resolved_path, ":") && ~startsWith(resolved_path, "files/")
    resolved_path = "files/" + erase(resolved_path, "./");
end

candidate_paths = strings(0, 1);
candidate_paths(end + 1) = resolved_path;
if strlength(mimic_root) > 0 && isfolder(char(mimic_root))
    candidate_paths(end + 1) = string(fullfile(char(mimic_root), strrep(char(resolved_path), '/', filesep)));
end

[folder_path, record_name, ~] = fileparts(char(resolved_path));
if isempty(record_name)
    [~, record_name] = fileparts(folder_path);
end

if mod(strlength(record_name), 2) == 0
    half_len = strlength(record_name) / 2;
    left_half = extractBefore(record_name, half_len + 1);
    right_half = extractAfter(record_name, half_len);
    if left_half == right_half
        collapsed_name = left_half;
        collapsed_rel = string(fullfile(folder_path, char(collapsed_name)));
        collapsed_rel = replace(collapsed_rel, "\\", "/");
        if strlength(mimic_root) > 0 && isfolder(char(mimic_root))
            candidate_paths(end + 1) = string(fullfile(char(mimic_root), strrep(char(collapsed_rel), '/', filesep)));
        end
        candidate_paths(end + 1) = collapsed_rel;
    end
end

candidate_paths = unique(candidate_paths, 'stable');

for i = 1:numel(candidate_paths)
    c = candidate_paths(i);
    c = regexprep(c, '/+$', '');
    hea_path = c + ".hea";
    dat_path = c + ".dat";
    if isfile(char(hea_path)) || isfile(char(dat_path))
        resolved_path = c;
        ok = true;
        return;
    end
end

message = sprintf('Local ECG record not found for "%s" (checked %d candidates).', char(record_path), numel(candidate_paths));
end
end


function features = extract_93_features(Amplitude_features, Timing_leadwise, Timing_sync, FPT_Cell, signal)
features = struct();

lead_idx = struct( ...
    'I', 1, 'II', 2, 'III', 3, ...
    'aVR', 4, 'aVL', 5, 'aVF', 6, ...
    'V1', 7, 'V2', 8, 'V3', 9, 'V4', 10, 'V5', 11, 'V6', 12 ...
);

Q_IDX = 2;
R_IDX = 3;
S_IDX = 4;
T_IDX = 5;

compute_stats = @(values) struct( ...
    'mean', mean(values, 'omitnan'), ...
    'count', sum(~isnan(values)), ...
    'iqr', iqr(values) ...
);

r_leads = {'I', 'II', 'III', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL'};
for i = 1:length(r_leads)
    lead = r_leads{i};
    idx = lead_idx.(lead);
    r_values = squeeze(Amplitude_features(idx, :, R_IDX));
    stats = compute_stats(r_values);
    features.(['R_Amp_' lead]) = stats.mean;
    features.(['R_Amp_' lead '_count']) = stats.count;
    features.(['R_Amp_' lead '_iqr']) = stats.iqr;
end

s_leads = {'V5', 'V6'};
for i = 1:length(s_leads)
    lead = s_leads{i};
    idx = lead_idx.(lead);
    s_values = squeeze(Amplitude_features(idx, :, S_IDX));
    stats = compute_stats(s_values);
    features.(['S_Amp_' lead]) = stats.mean;
    features.(['S_Amp_' lead '_count']) = stats.count;
    features.(['S_Amp_' lead '_iqr']) = stats.iqr;
end

t_leads = {'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'};
for i = 1:length(t_leads)
    lead = t_leads{i};
    idx = lead_idx.(lead);
    t_values = squeeze(Amplitude_features(idx, :, T_IDX));
    stats = compute_stats(t_values);
    features.(['T_Amp_' lead]) = stats.mean;
    features.(['T_Amp_' lead '_count']) = stats.count;
    features.(['T_Amp_' lead '_iqr']) = stats.iqr;
end

FPT_mat = cat(3, FPT_Cell{:});
st_leads = {'II', 'III', 'aVR', 'aVF'};
for i = 1:length(st_leads)
    lead = st_leads{i};
    idx = lead_idx.(lead);
    t_onsets = squeeze(FPT_mat(:,10,idx));
    s_offsets = squeeze(FPT_mat(:,8,idx));
    st_values = nan(size(t_onsets));
    for b = 1:length(t_onsets)
        t_on = t_onsets(b);
        s_off = s_offsets(b);
        if ~isnan(t_on) && ~isnan(s_off)
            t_on = max(1, min(size(signal,1), round(t_on)));
            s_off = max(1, min(size(signal,1), round(s_off)));
            st_values(b) = signal(t_on, idx) - signal(s_off, idx);
        end
    end
    stats = compute_stats(st_values);
    features.(['ST_Elev_' lead]) = stats.mean;
    features.(['ST_Elev_' lead '_count']) = stats.count;
    features.(['ST_Elev_' lead '_iqr']) = stats.iqr;
end

if ~isempty(Timing_sync)
    rr_vals = Timing_sync(:,8);
    stats = compute_stats(rr_vals);
    features.RR_Mean_Global = stats.mean;
    features.RR_Mean_Global_count = stats.count;
    features.RR_Mean_Global_iqr = stats.iqr;

    qrs_vals = Timing_sync(:,2);
    stats = compute_stats(qrs_vals);
    features.QRS_Dur_Global = stats.mean;
    features.QRS_Dur_Global_count = stats.count;
    features.QRS_Dur_Global_iqr = stats.iqr;

    qtc_vals = Timing_sync(:,7);
    stats = compute_stats(qtc_vals);
    features.QT_IntFramingham_Global = stats.mean;
    features.QT_IntFramingham_Global_count = stats.count;
    features.QT_IntFramingham_Global_iqr = stats.iqr;

    pr_vals = Timing_sync(:,5);
    stats = compute_stats(pr_vals);
    features.PR_Int_Global = stats.mean;
    features.PR_Int_Global_count = stats.count;
    features.PR_Int_Global_iqr = stats.iqr;
end
end
