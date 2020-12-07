%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('./Utils'));
fprintf('Add path done !!\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PATH = './testdata/Test_set_light';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist(path, 'dir')
   mkdir(path)
end

% Load the raw data
load('./dataset/raw_data_test.mat')

D = size(Table.light, 2);
gap = 48/D;
t = (0:D-1)*48/D;

load('elevation.mat')
ele_long_grid = [-130:1:-66];
ele_lat_grid = [10:1:54];

NUM = height(Table);
DISPLAY = 0;
ALT = 1;

% Sampling rate: 1 minute
time_unit = 1/60;    

% Range to pay attention, defined manually 
WIN_LEN = 4;
WIN_RANGE = [-WIN_LEN/2:time_unit:WIN_LEN/2-time_unit];

range_long = [-10:1:10];
range_lat = [-30:1:30];
long_num = length(range_long);
lat_num = length(range_lat);


for i = 1:NUM
    
    fprintf('Processing #%d\n', i);
    year = Table.year(i);
    month = Table.month(i);
    day = Table.day(i);
    [month_next, day_next] = next_day(month, day, 1);
    
    long = Table.longitude(i);
    lat = Table.latitude(i);
    if ALT == 1
        alt = Table.altitude(i);
    else
        alt = 0;
    end
    shift = Table.shift(i);
    
    long_grid = round(long) + range_long;
    lat_grid = round(lat) + range_lat;
    
    test_light = zeros(long_num, lat_num, WIN_LEN*2/time_unit);
    light_mask = zeros(long_num, lat_num);
                
    for m = 1:long_num
        for n = 1:lat_num
            longitude = long_grid(m);
            latitude = lat_grid(n);
            
            % Check whether it is out of range
            if longitude <-130 || longitude > -66 || latitude < 10 || latitude > 54
                light_intensity= -1000*ones(D/48*8,1);
                light_mask(m,n) = 1;
            else
                % Find out the true altitude at that particular location
                alt = interp2(ele_long_grid,ele_lat_grid,elem,[longitude],[latitude], 'linear');
                [sunrise_prev, sunset_prev] = get_sun_data_offline(latitude, longitude, alt, year, month, day);
                [sunrise_next, sunset_next] = get_sun_data_offline(latitude, longitude, alt, year, month_next, day_next);
                
                if sunrise_next == -1000 || sunset_next== -1000 || sunrise_prev == -1000 || sunset_prev == -1000
                    light_intensity = -1000*ones(D/48*8,1);
                    light_mask(m,n) = 1;
                else
                    intensity = Table.light(i,:);
                    % Get ground truth shifted time
                    sunset_prev = sunset_prev - shift*gap;
                    sunrise_next = sunrise_next - shift*gap + 24;
    
                    night_len = sunrise_next - sunset_prev;
                    
                    scale = 12/night_len;
 
                    focus_prev = WIN_RANGE/scale + sunset_prev;
                    focus_next = WIN_RANGE/scale + sunrise_next;
                    focus = [focus_prev, focus_next];
                         
                    curve_out = interp1(t, intensity, focus, 'linear');
                    test_light(m,n,:) = curve_out;
                end
            end
        end
    end
        
    save([PATH,'/',num2str(i),'.mat'], 'test_light', 'light_mask');
end