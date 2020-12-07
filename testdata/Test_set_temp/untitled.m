
for i = 1:781
    load([num2str(i), '.mat'])
    test_temp = test_temp(:,:,[12-8:12+7, 36-8:36+7]);
    save([num2str(i), '.mat'], 'test_temp', 'temp_mask')
end
            