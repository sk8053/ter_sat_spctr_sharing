clear;
subplot(1,2,1);
img = imread("rural_scenario.png");
imshow(img);

subplot(1,2,2);
snr = importdata("vis_sat_n.txt");
plot(snr, 'k*-.');
grid on;