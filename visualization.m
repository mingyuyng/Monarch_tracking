close all;
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('./Utils'));
fprintf('Add path done !!\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HEATMAP_LIGHT = './results/Heatmaps_light/';
HEATMAP_TEMP = './results/Heatmaps_temp/';
PATH_LIGHT = './testdata/Test_set_light/';
PATH_TEMP = './testdata/Test_set_temp/';
SAVEFIG = 0;
PATH_FIG = './results/heatmap_visual';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./testdata/stats.mat')
if ~exist(PATH_FIG, 'dir')
   mkdir(PATH_FIG)
end

D = 5760;
gap = 48/D;
t = (0:D-1)*48/D;

NUM = 20;

for i = 1:NUM
    
    long_cor = -10:1:10;
    lat_cor = -30:1:30;
    [long_cor_grid,lat_cor_grid] = meshgrid(long_cor,lat_cor);
    
    long_fine = -10:0.1:10;
    lat_fine = -30:0.1:30;
    [long_fine_grid,lat_fine_grid] = meshgrid(long_fine,lat_fine);
    
    load([PATH_LIGHT,num2str(i),'.mat'])
    load([PATH_TEMP,num2str(i),'.mat'])
    
    % Read the coarse heatmap for both light and temperature
    load([HEATMAP_LIGHT,'light_',num2str(i),'.mat']);
    result_light = results';
    heatmap_light = result_light.*(1-light_mask');
    load([HEATMAP_TEMP, 'temp_',num2str(i),'.mat']);
    result_temp = results';
    heatmap_temp = result_temp.*(1-temp_mask');
    
    % Do coarse estimation on light hetmap
    [max_cor,max_idx] = max(heatmap_light(:));
    [lat_idx, long_idx]=ind2sub(size(heatmap_light),max_idx);
    long_light_coarse = long_cor(long_idx);
    lat_light_coarse = lat_cor(lat_idx);
    
    % Combine two heatmaps and do coarse estimation
    heatmap_comb = heatmap_temp .* heatmap_light;
    [max_cor,max_idx] = max(heatmap_comb(:));
    [lat_idx, long_idx]=ind2sub(size(heatmap_comb),max_idx);
    long_comb_coarse = long_cor(long_idx);
    lat_comb_coarse = lat_cor(lat_idx);
    
    % Calculate the fine heatmap
    heatmap_light_fine = interp2(long_cor_grid,lat_cor_grid,heatmap_light,long_fine_grid,lat_fine_grid, 'cubic',0);
    heatmap_temp_fine = interp2(long_cor_grid,lat_cor_grid,heatmap_temp,long_fine_grid,lat_fine_grid, 'cubic',0);
    heatmap_light_fine(heatmap_light_fine<0) = 0;
    heatmap_temp_fine(heatmap_temp_fine<0) = 0;
    
    heatmap_combined = heatmap_light_fine .* heatmap_temp_fine;
    
    
    if SAVEFIG == 1
        
        figure('visible', 'off');
        subplot(2,3,1)
        title(['month:',num2str(month(i)),'  day:', num2str(day(i)), ' LIGHT'])
        surface(long_fine_grid, lat_fine_grid, heatmap_light_fine,'edgecolor', 'None');hold on
        ylim([-10,10])
        xlim([-10,10])
        xlabel('longitude')
        ylabel('latitude')
        
        scatter3(long_light_coarse, lat_light_coarse, 1, 20, 'k', 'filled')
        
        subplot(2,3,2)
        title(['month:',num2str(month(i)),'  day:', num2str(day(i)), ' TEMP'])
        surface(long_fine_grid,lat_fine_grid,heatmap_temp_fine,'edgecolor', 'None');hold on
        ylim([-10,10])
        xlim([-10,10])
        xlabel('longitude')
        ylabel('latitude')
        
        subplot(2,3,3)
        title(['month:',num2str(month(i)),'  day:', num2str(day(i)), ' COMBINED'])
        surface(long_fine_grid,lat_fine_grid,heatmap_combined,'edgecolor', 'None');hold on
        ylim([-10,10])
        xlim([-10,10])
        xlabel('longitude')
        ylabel('latitude')
        
        scatter3(long_comb_coarse, lat_comb_coarse, 1, 20, 'k', 'filled')
        
        
        subplot(2,3,4)
        lights = [];
        for j = 1:21
            mm = heatmap_light(:,j) > 0.3;
            if sum(mm)>1
                lights = [lights;squeeze(test_light(j,mm,:))];
            elseif sum(mm)==1
                lights = [lights;squeeze(test_light(j,mm,:))'];
            end
        end
        
        plot(lights'); hold on
        title('light curves with high prob')
        plot(120*ones(1,100), linspace(0,2,100), 'linewidth',2);hold on
        plot(360*ones(1,100), linspace(0,2,100), 'linewidth',2);hold on
        ylim([0,4])
        
        subplot(2,3,5)
        temps = [];
        for j = 1:21
            mm = heatmap_temp(:,j) > 0.3;
            if sum(mm)>1
                temps = [temps;squeeze(test_temp(j,mm,:))];
            elseif sum(mm)==1
                temps = [temps;squeeze(test_temp(j,mm,:))'];
            end
        end
        
        plot(temps'); hold on
        title('temperature curves with high prob')
        ylim([-10,30])
        
        saveas(gcf, [PATH_FIG, '/', num2str(i), '.png'])
        close()
    end
    
    fprintf('Processing #%d\n', i)
end
