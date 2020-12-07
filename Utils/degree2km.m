function [err_out] = degree2km(long_true,lat_true,long_est,lat_est)

N = length(long_true);
err_out = zeros(N,1);

for i = 1:N
    r = 6371;
    long_t = long_true(i)/180*pi;
    long_e = long_est(i)/180*pi;
    lat_t = lat_true(i)/180*pi;
    lat_e = lat_est(i)/180*pi;

    sigma = sin(lat_t)*sin(lat_e) + cos(lat_t)*cos(lat_e)*cos(abs(long_e-long_t));

    %delta_x = cos(long_true)*cos(lat_true) - cos(long_est)*cos(lat_est);
    %delta_y = cos(long_true)*sin(lat_true) - cos(long_est)*sin(lat_est);
    %delta_z = sin(long_true) - sin(long_est);

    %C = sqrt(delta_x^2 + delta_y^2 + delta_z^2);

    %err_km = r*2*asin(C/2);

    err_km = r*acos(sigma);
    err_out(i) = err_km;
end


