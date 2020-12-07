clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('./Utils'));
fprintf('Add path done !!\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the raw data
load('./dataset/raw_data_train.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DISPLAY = 0;    % Set to 1 if you want some visualization
ALT = 1;        % Set to 1 if altitude information is included
NAME_TRAIN = './dataset/Light_train_8.mat';
NAME_VALID = './dataset/Light_valid_8.mat';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = size(Table.light, 2);
gap = 48/D;
t = (0:D-1)*48/D;

% Standard curves are defined with a sunset within 18:00+st_rng and a sunrise
% within 6:00+st_rng

% Non-standard curves are defined with a sunset within 18:00+nonst_rng and a sunrise
% within 6:00+nonst_rng

% Unit: minutes
nonst_rng = [-60:-1,1:60];    
st_rng = [-0.5:0.1:0.5];   

% Weights for weighted sampling, different for night center and night
% length
w_cen = exp(([-60:-1]+1)/100);
w_cen = [w_cen, w_cen(end:-1:1)];

% For each volunteer light curve, we generate N_true standard curves and
% N_false non-standard curves
N_true = 5;
N_false = 15;

% Final sampling rate for the Neural Network: 1 minute
time_unit = 1/60;    

% Range to pay attention, defined manually 
WIN_LEN = 4;
WIN_RANGE = [-WIN_LEN/2:time_unit:WIN_LEN/2-time_unit];

INDEX_TRAIN = find(Table.valid==0);
INDEX_VALID = find(Table.valid==1); 

light_train = zeros(length(INDEX_TRAIN)*(N_true+N_false), WIN_LEN*2/time_unit);
target_train = zeros(length(INDEX_TRAIN)*(N_true+N_false), 1);
light_valid = zeros(length(INDEX_VALID)*(N_true+N_false), WIN_LEN*2/time_unit);
target_valid = zeros(length(INDEX_VALID)*(N_true+N_false), 1);

% Summary of dataset
fprintf('Training set size: %d, class 0: %d, class 1: %d \n', length(INDEX_TRAIN)*(N_true+N_false), length(INDEX_TRAIN)*N_false, length(INDEX_TRAIN)*N_true);
fprintf('Validation set size: %d, class 0: %d, class 1: %d \n', length(INDEX_VALID)*(N_true+N_false), length(INDEX_VALID)*N_false, length(INDEX_VALID)*N_true);

fprintf('Generating Training...\n');


% Generate Training set
for i = 1:length(INDEX_TRAIN)
     
    index = INDEX_TRAIN(i);
    
    % Read the basic information of the light observation
    year = Table.year(index);
    month = Table.month(index);
    day = Table.day(index);
    [month_next, day_next] = next_day(month, day, 1);
    long = Table.longitude(index);
    lat = Table.latitude(index);
    shift = Table.shift(index);
    light_intensity = Table.light(index,:);
    if ALT == 1
        alt = Table.altitude(index);
    else
        alt = 0;
    end
    
    % Get ground truth sunrise and sunset time 
    [sunrise_prev, sunset_prev] = get_sun_data_offline(lat, long, alt, year, month, day);
    [sunrise_next, sunset_next] = get_sun_data_offline(lat, long, alt, year, month_next, day_next);
    sunset_prev = sunset_prev - shift*gap;
    sunrise_next = sunrise_next - shift*gap + 24; 
    
    %%%%%%%%%%%%%%%%%%%%%% Generate Positive data %%%%%%%%%%%%%%%%%%%%%%%%%
    for num = 1:N_true
            
       count = (i-1)*(N_false+N_true) + num;
       
       % Sample within the range
       sunset_prev_t = sunset_prev + datasample(st_rng,1)/60;
       sunrise_next_t = sunrise_next + datasample(st_rng,1)/60;
       night_len_t = sunrise_next_t-sunset_prev_t;
       
       scale = 12/night_len_t;
       focus_prev = WIN_RANGE/scale + sunset_prev_t;
       focus_next = WIN_RANGE/scale + sunrise_next_t;
       focus = [focus_prev, focus_next];
       
       curve_out = interp1(t, light_intensity, focus, 'linear', 0);
       
       if DISPLAY == 1
            plot([18+WIN_RANGE, 30+WIN_RANGE], curve_out, 'color', 'b'); grid on; hold on; drawnow;
            ylim([0,4])
       end
       
       light_train(count, :) = curve_out;
       target_train(count) = 1;
            
       if mod(count, 100) ==0 
           count
       end    
             
    end
    
    if DISPLAY == 1
        plot(18*ones(1,100), linspace(0,4,100), '--', 'color', 'k', 'linewidth', 2);
        plot(30*ones(1,100), linspace(0,4,100), '--', 'color', 'k', 'linewidth', 2);
        plot(18+WIN_LEN/2*ones(1,100), linspace(0,4,100), 'color', 'r', 'linewidth', 2);
        plot(30-WIN_LEN/2*ones(1,100), linspace(0,4,100), 'color', 'r', 'linewidth', 2);
        plot(18-WIN_LEN/2*ones(1,100), linspace(0,4,100), 'color', 'r', 'linewidth', 2);
        plot(30+WIN_LEN/2*ones(1,100), linspace(0,4,100), 'color', 'r', 'linewidth', 2);
        ylim([0,4])
        xlabel('time')
        ylabel('pre-processed light intensity: log10')
        title(['standard curves from light ', num2str(index)])
        hold off
        figure;
    end
    
    for num = 1:N_false
        
        count = (i-1)*(N_false+N_true) + N_true + num;
        
        sunset_prev_t = sunset_prev + datasample(nonst_rng,1,'Weights',w_cen)/60;
        sunrise_next_t = sunrise_next + datasample(nonst_rng,1,'Weights',w_cen)/60;
       
        night_len_t = sunrise_next_t-sunset_prev_t;
       
        scale = 12/night_len_t;
 
        focus_prev = WIN_RANGE/scale + sunset_prev_t;
        focus_next = WIN_RANGE/scale + sunrise_next_t;
        focus = [focus_prev, focus_next];
            
        curve_out = interp1(t, light_intensity, focus, 'linear', 0);    
        light_train(count, :) = curve_out;
        target_train(count) = 0;
        
        if DISPLAY == 1
            plot([18+WIN_RANGE, 30+WIN_RANGE], curve_out, 'color', 'b'); grid on; hold on; drawnow;
            ylim([0,4])
        end
        
        if mod(count, 100) ==0
            count
        end    
    end
    
    if DISPLAY == 1
        plot(18*ones(1,100), linspace(0,4,100), '--', 'color', 'k', 'linewidth', 2);
        plot(30*ones(1,100), linspace(0,4,100), '--', 'color', 'k', 'linewidth', 2);
        plot(18+WIN_LEN/2*ones(1,100), linspace(0,4,100), 'color', 'r', 'linewidth', 2);
        plot(30-WIN_LEN/2*ones(1,100), linspace(0,4,100), 'color', 'r', 'linewidth', 2);
        plot(18-WIN_LEN/2*ones(1,100), linspace(0,4,100), 'color', 'r', 'linewidth', 2);
        plot(30+WIN_LEN/2*ones(1,100), linspace(0,4,100), 'color', 'r', 'linewidth', 2);
        ylim([0,4])
        xlabel('time')
        ylabel('pre-processed light intensity: log10')
        title(['non-standard curves from light ', num2str(index)])
        hold off
    end
    
      
end


fprintf('Generating Validation set...');

% Generate Training set
for i = 1:length(INDEX_VALID)
    
    index = INDEX_VALID(i);
    
    year = Table.year(index);
    month = Table.month(index);
    day = Table.day(index);
    [month_next, day_next] = next_day(month, day, 1);
    
    long = Table.longitude(index);
    lat = Table.latitude(index);
    if ALT == 1
        alt = Table.altitude(index);
    else
        alt = 0;
    end
    shift = Table.shift(index);
    
    light_intensity = Table.light(index,:);
    % Get ground truth time 
    [sunrise_prev, sunset_prev] = get_sun_data_offline(lat, long, alt, year, month, day);
    [sunrise_next, sunset_next] = get_sun_data_offline(lat, long, alt, year, month_next, day_next);
    
    % Get ground truth shifted time
    sunset_prev = sunset_prev - shift*gap;
    sunrise_next = sunrise_next - shift*gap + 24;
    
    night_len = sunrise_next - sunset_prev;
       
    %%%%%%%%%%%%%%%%%%%%%% Generate Positive data %%%%%%%%%%%%%%%%%%%%%%%%%
    for num = 1:N_true
            
       count = (i-1)*(N_false+N_true) + num;
       
       sunset_prev_t = sunset_prev + datasample(st_rng,1)/60;
       sunrise_next_t = sunrise_next + datasample(st_rng,1)/60;
       
       night_len_t = sunrise_next_t-sunset_prev_t;
       
       scale = 12/night_len_t;
 
       focus_prev = WIN_RANGE/scale + sunset_prev_t;
       focus_next = WIN_RANGE/scale + sunrise_next_t;
       focus = [focus_prev, focus_next];
       
       curve_out = interp1(t, light_intensity, focus, 'linear', 0);
       light_valid(count, :) = curve_out;
       target_valid(count) = 1;
            
       if mod(count, 100) ==0 
           count
       end    
             
    end
    
    for num = 1:N_false
        
        count = (i-1)*(N_false+N_true) + N_true + num;
        
        sunset_prev_t = sunset_prev + datasample(nonst_rng,1,'Weights',w_cen)/60;
        sunrise_next_t = sunrise_next + datasample(nonst_rng,1,'Weights',w_cen)/60;
       
        night_len_t = sunrise_next_t-sunset_prev_t;
       
        scale = 12/night_len_t;
 
        focus_prev = WIN_RANGE/scale + sunset_prev_t;
        focus_next = WIN_RANGE/scale + sunrise_next_t;
        focus = [focus_prev, focus_next];
            
        curve_out = interp1(t, light_intensity, focus, 'linear', 0);    
        light_valid(count, :) = curve_out;
        target_valid(count) = 0;
       
        if mod(count, 100) ==0
            count
        end    
    end
      
end


% Save the training data and validation data
data = light_train;
label = target_train;
save(NAME_TRAIN, 'data', 'label');

data = light_valid;
label = target_valid;
save(NAME_VALID, 'data', 'label');




