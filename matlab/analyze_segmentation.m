% Analyze segmentation masks dumped by the Python app.
% Set run_dir to a folder inside runs/ created by --dump.

run_dir = fullfile('..', 'runs', 'YYYYMMDD_HHMMSS'); % TODO: change

frame_files = dir(fullfile(run_dir, 'frame_*.png'));
mask_files = dir(fullfile(run_dir, 'mask_*.png'));

if isempty(frame_files) || isempty(mask_files)
    error('No frames/masks found in %s', run_dir);
end

idx = 1; % pick a sample index
frame = imread(fullfile(run_dir, frame_files(idx).name));
mask = imread(fullfile(run_dir, mask_files(idx).name));

mask_f = double(mask) / 255.0;

figure; subplot(2,3,1); imshow(frame); title('Frame');
subplot(2,3,2); imshow(mask_f); title('Mask (soft)');
subplot(2,3,3); imhist(mask); title('Histogram mask');

% Otsu thresholding as a baseline (compare with fixed threshold in Python)
level = graythresh(mask_f);
bw = imbinarize(mask_f, level);

% Morphological refinement (open/close)
se = strel('disk', 3);
bw_open = imopen(bw, se);
bw_close = imclose(bw_open, se);

subplot(2,3,4); imshow(bw); title('Otsu binarize');
subplot(2,3,5); imshow(bw_close); title('After morph');

% Overlay segmentation on the frame
overlay = frame;
mask_rgb = cat(3, bw_close, bw_close, bw_close);
overlay(mask_rgb) = overlay(mask_rgb) * 0.3 + 255 * 0.7;
subplot(2,3,6); imshow(overlay); title('Overlay');

% Simple stats
area_pixels = sum(bw_close(:));
ratio = area_pixels / numel(bw_close);
fprintf('Foreground ratio: %.3f\n', ratio);
