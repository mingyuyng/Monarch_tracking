function [output] = is_end_month(m,d)
%IS_END_MONTH �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
if (m == 1 || m == 3 || m == 5 || m == 7 || m == 8 || m == 10 || m == 12) && d == 31
    output = 1;
elseif (m == 4 || m == 6 || m == 9 || m == 11) && d == 30
    output = 1;
elseif m == 2 && d == 28
    output = 1;
else
    output = 0;
end

