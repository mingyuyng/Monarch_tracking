function [month,day] = next_day(mm,dd,num_day)
%NEXT_DAY This function calculates the date in a small amount of time
%   此处显示详细说明

if (mm == 1 || mm == 3 || mm == 5 || mm == 7 || mm == 8 || mm == 10 || mm == 12) && (dd + num_day) > 31
    month = mm + 1;
    day = dd + num_day - 31;
elseif (mm == 4 || mm == 6 || mm == 9 || mm == 11) && (dd + num_day) > 30
    month = mm + 1;
    day = dd + num_day - 30;
elseif mm == 2 && (dd + num_day) > 28
    month = mm + 1;
    day = dd + num_day - 28;
else
    month = mm;
    day = dd + num_day;
end

