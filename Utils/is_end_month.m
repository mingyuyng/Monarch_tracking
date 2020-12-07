function [output] = is_end_month(m,d)
%IS_END_MONTH 此处显示有关此函数的摘要
%   此处显示详细说明
if (m == 1 || m == 3 || m == 5 || m == 7 || m == 8 || m == 10 || m == 12) && d == 31
    output = 1;
elseif (m == 4 || m == 6 || m == 9 || m == 11) && d == 30
    output = 1;
elseif m == 2 && d == 28
    output = 1;
else
    output = 0;
end

