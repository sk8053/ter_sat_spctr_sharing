clear;
n_ant = 8;

G_T = 13;
EK = -198.6;
BW = 30e6;
bs_tx_power = 33;

inr_nlos_nulling_1 = importdata('data_new/downlink_inr_null_nlos_1.txt');
inr_nlos_nulling_0 = importdata('data_new/downlink_inr_null_nlos_0.txt');

inr_los_nulling_1 = importdata('data_new/downlink_inr_null_los_1.txt');
inr_los_nulling_0 = importdata('data_new/downlink_inr_null_los_0.txt');
inr_svd = importdata('data_new/downlink_inr_SVD_0.txt');

loss_nlos_0 = importdata('data_new/delta_null_nlos_0.txt');
loss_nlos_1 = importdata('data_new/delta_null_nlos_1.txt');

loss_los_0 = importdata('data_new/delta_null_los_0.txt');
loss_los_1 = importdata('data_new/delta_null_los_1.txt');
loss_svd = importdata('data_new/delta_SVD_0.txt');


set(gca,'fontname','times new roman');  
lw = 2.5;
t = tiledlayout(1,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';

h1 = cdfplot(inr_svd);
hold on;

h2 = cdfplot(inr_los_nulling_0);
hold on;
h3 = cdfplot(inr_los_nulling_1);
hold on;

h4 = cdfplot(inr_nlos_nulling_0);
hold on;
h5 = cdfplot(inr_nlos_nulling_1);
hold on;

set(gca,'fontname','times new roman');  
set(h1, 'LineWidth', 2.5, 'LineStyle','-');
set(h1, 'Color', 'k');

set(h2, 'LineWidth',1.5);
set(h2, 'Color', 'blue');
set(h2, 'LineStyle','-.');

set(h3, 'LineWidth', 3.5);
h3.Color =  "blue";
set(h3, 'LineStyle','-.');

set(h4, 'LineWidth', 1.5);
h4.Color =  "red";
set(h4, 'LineStyle',':', 'LineWidth',1.5);


set(h5, 'LineWidth', 3.5);
set(h5, 'Color', 'red');
set(h5, 'LineStyle',':');

xlabel('INR [dB]','Interpreter','latex','fontsize', 13);
ylabel('CDF','Interpreter','latex','fontsize', 13);
xticks(linspace(-40,10,6));
yticks(linspace(0,1,11))

xline(-6, 'k:', 'LineWidth',1.5);
title ('')
ax = gca;
ax.GridLineWidth = 2;
set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));


xlim([-25,12]);
set(gca,'fontname','times new roman'); 
set(gcf,'Position',[100 100 500 300]);
exportgraphics(gcf,'figures/inr_cdf.png', 'Resolution',600);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure();
set(gca,'fontname','times new roman');
t = tiledlayout(1,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
nexttile;

h1 = cdfplot(loss_svd);
hold on;

h2 = cdfplot(loss_los_0);
hold on;
h3 = cdfplot(loss_los_1);
hold on;

h4 = cdfplot(loss_nlos_0);
hold on;
h5 = cdfplot(loss_nlos_1);
hold on;

set(h1, 'LineWidth', 2.5);
set(h1, 'Color', 'k');

set(h2, 'LineWidth', 1.5);
set(h2, 'Color', 'b');
set(h2, 'LineStyle','-.');

set(h3, 'LineWidth', 3.5);
h3.Color = "b";
set(h3, 'LineStyle','-.');

set(h4, 'LineWidth', 1.5);
h4.Color = "red";
set(h4, 'LineStyle',':');

set(h5, 'LineWidth', 3.5);
h5.Color = "red";
set(h5, 'LineStyle',':');
xline(3, 'k:', 'LineWidth',1.5);

legend([h1 h2 h3,h4,h5], 'no nulling', ..., 
    'LOS nulling, $\lambda=1$', ...
    'LOS nulling, $\lambda=10$', ...,
    'multi-path nulling, $\lambda=1$ ', ...,
    'multi-path nulling, $\lambda=10$', ...,
    'Location', 'southeast', 'interpreter', 'latex', 'fontsize', 12);
grid on;

ax = gca;
ax.GridLineWidth = 2;
set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));
title('')
grid on;
xlim([0,3.5]);

xticks(linspace(0, 3.5,8));
yticks(linspace(0,1,11))
lg  = legend('Orientation','Horizontal','NumColumns',1); 

xlabel('Loss [dB]','Interpreter','latex','fontsize', 13);
set(gca,'fontname','times new roman');  
ylabel('CDF','Interpreter','latex','FontSize',13);
set(gcf,'Position',[100 100 500 300]);

exportgraphics(gcf,'figures/loss_cdf.png', "Resolution",600);

