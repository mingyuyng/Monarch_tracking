function [curve_out] = reshape_curve_double_new(curve_input,set_input,rise_input,set_tar,rise_tar,time_unit,window)
%SHIFT_CURVE reshape the light curve according to the target day center and
%day length
%   center: MATLAB function 'circshift'
%   length: MATLAB function 'resample'

signal_len = length(curve_input);
t = (0:signal_len-1)*time_unit;

% 1. Shift the light curve to 24:00 (center)
center_input = (rise_input+set_input)/2;
shift_center = round((24-center_input)/time_unit);
light_intensity = circshift(curve_input,shift_center);
set_input = set_input + shift_center*time_unit;
rise_input = rise_input + shift_center*time_unit;

%plot(t,light_intensity); hold on
%plot(24*ones(1,100),linspace(0,5,100));hold on

% 2. Resample to change the day length
length_tar = rise_tar - set_tar;
length_input = rise_input - set_input;
rate = length_tar/length_input; 
resample_rate = round(rate*signal_len/2)*2;

curve_scaled = resample(light_intensity, resample_rate, signal_len);

%plot(t,light_intensity); hold on


% 3. Shift the light curve to the desired center

center_tar = (rise_tar+set_tar)/2;
shift_final = round((center_tar-24)/time_unit);
light_intensity_new = circshift(curve_scaled,shift_final);

win_size = 20/time_unit;
light_intensity_new = light_intensity_new(length(light_intensity_new)/2-win_size/2:length(light_intensity_new)/2+win_size/2-1);

curve_out = light_intensity_new(round((window-14)/time_unit));

%if length(curve_out) ~= len_edge
%    keyboard
%end

end



