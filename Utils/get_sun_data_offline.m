function [rise_hour,set_hour] = get_sun_data_offline(Latitude, Longitude, lat, yy, mm, dd)
%GET_SUN_DATA_OFFLINE: get the corresponding sunrise & sunset data offline
%   input: Latitude, Longitude, year, month, day 
    
    [rise, set] = sunrise(Latitude,Longitude,lat,0,[yy,mm,dd]);
    
    if imag(rise) ~= 0 || imag(set) ~= 0
        rise_hour = -1000;
        set_hour = -1000;
    else
        rise = datetime(rise,'ConvertFrom','datenum');
        set  = datetime(set,'ConvertFrom','datenum');
    
        rise_hour = hour(rise) + minute(rise)/60 + second(rise)/3600;
        set_hour  = hour(set)  + minute(set)/60  + second(set)/3600;
        
        if set_hour < rise_hour
           set_hour = set_hour + 24;
        end
    end
    
    
    
end

