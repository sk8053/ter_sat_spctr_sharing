clear;
n_ant = 8;

G_T = 13;
EK = -198.6;
BW = 30e6;
bs_tx_power = 33;

inr_nlos_nulling_1 = importdata('data_new/downlink_inr_null_nlos_1_mumimo.txt');
inr_nlos_nulling_0 = importdata('data_new/downlink_inr_null_nlos_0_mumimo.txt');

inr_los_nulling_1 = importdata('data_new/downlink_inr_null_los_1_mumimo.txt');
inr_los_nulling_0 = importdata('data_new/downlink_inr_null_los_0_mumimo.txt');
inr_svd = importdata('data_new/downlink_inr_SVD_0.txt');

%inr_nlos_nulling_1 = 10*log10(itf_nlos_nulling_1) + G_T - EK -10*log10(BW) + bs_tx_power;
%inr_nlos_nulling_0 = 10*log10(itf_nlos_nulling_0) + G_T - EK -10*log10(BW) + bs_tx_power;

%inr_los_nulling_1 = 10*log10(itf_los_nulling_1) + G_T - EK -10*log10(BW) + bs_tx_power;
%inr_los_nulling_0 = 10*log10(itf_los_nulling_0) + G_T - EK -10*log10(BW) + bs_tx_power;
%inr_svd = 10*log10(itf_svd) + G_T - EK -10*log10(BW) + bs_tx_power;


%loss10_cbook = importdata('data8t8/GainLoss_list_12GHz_codebook_10.txt');
loss_nlos_0 = importdata('data_new/delta_null_nlos_0_mumimo.txt');
loss_nlos_1 = importdata('data_new/delta_null_nlos_1_mumimo.txt');

loss_los_0 = importdata('data_new/delta_null_los_0_mumimo.txt');
loss_los_1 = importdata('data_new/delta_null_los_1_mumimo.txt');
loss_svd = importdata('data_new/delta_SVD_0.txt');

%loss_los_0 = 10*log10(loss_los_0);
%loss_los_1 = 10*log10(loss_los_1);
%loss_nlos_0 = 10*log10(loss_nlos_0);
%loss_nlos_1 = 10*log10(loss_nlos_1);
%loss_svd = 10*log10(loss_svd);

set(gca,'fontname','times new roman');  
lw = 2.5;
t = tiledlayout(1,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
%subplot(1,2,1);
%nexttile;
%figure();
h1 = cdfplot(inr_svd);
hold on;
%h2 = cdfplot(inr10_cbook);
%hold on;

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

%set(h6, 'LineWidth', lw);
%set(h6, 'Color', 'b');
%set(h6, 'LineStyle','-.','LineWidth',2.5);

%title('SNR','fontsize', 13);
xlabel('INR [dB]','Interpreter','latex','fontsize', 13);
ylabel('CDF','Interpreter','latex','fontsize', 13);
xticks(linspace(-40,10,6));
yticks(linspace(0,1,11))

xline(-6, 'k:', 'LineWidth',1.5);
%title ('INR at satellites', fontsize = 13)
title ('')
ax = gca;
ax.GridLineWidth = 2;
set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));


xlim([-35,18]);
set(gca,'fontname','times new roman'); 
set(gcf,'Position',[100 100 500 300]);
exportgraphics(gcf,'figures/inr_cdf.png', 'Resolution',600);
%exportgraphics(gcf,'figures/inr_cdf.eps');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%figure();
set(gca,'fontname','times new roman');
t = tiledlayout(1,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
nexttile;
%subplot(1,2,1);

h1 = cdfplot(loss_svd);
hold on;
%h2 = cdfplot(loss10_cbook);
%hold on;

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

%set(gca, 'XTick', -10:10:43);
ax = gca;
ax.GridLineWidth = 2;
set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));
%title ('SNR loss at terrestrial UEs', fontsize = 13)
title('')
grid on;
%lg  = legend('Orientation','Horizontal','NumColumns',1); 
%lg.Layout.Tile = 'East';
xlim([0,4]);

xticks(0:6)
yticks(linspace(0,1,11))
lg  = legend('Orientation','Horizontal','NumColumns',1); 
%lg.Layout.Tile = 'South';

xlabel('Loss [dB]','Interpreter','latex','fontsize', 13);
set(gca,'fontname','times new roman');  
ylabel('CDF','Interpreter','latex','FontSize',13);
set(gcf,'Position',[100 100 500 300]);

exportgraphics(gcf,'figures/loss_cdf.png', "Resolution",600);
 %exportgraphics(gcf,'figures/loss_cdf.eps');

