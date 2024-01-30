clear;
inr10_cbook = importdata('data/downlink_inr_12GHz_10_codebook.txt');
inr10_svd = importdata('data/downlink_inr_12GHz_10_SVD.txt');
inr10_cbf_ideal = importdata('data/downlink_inr_12GHz_10_CBF_12.txt');
inr10_cbf_tr = importdata('data/downlink_inr_12GHz_10_CBF2_12.txt');

snr10_cbook = importdata('data/SNR_list_12GHz_codebook_10.txt');
snr10_svd = importdata('data/SNR_list_12GHz_SVD_10.txt');
snr10_cbf_ideal = importdata('data/SNR_list_12GHz_CBF_10_12.txt');
snr10_cbf_tr = importdata('data/SNR_list_12GHz_CBF2_10_12.txt');

set(gca,'fontname','times new roman');  

t = tiledlayout(1,2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
%subplot(1,2,1);
nexttile;
h1 = cdfplot(inr10_cbook);
hold on;
h2 = cdfplot(inr10_svd);
hold on;

h3 = cdfplot(inr10_cbf_tr);
hold on;
h4 = cdfplot(inr10_cbf_ideal);
hold on;

set(gca,'fontname','times new roman');  
set(h1, 'LineWidth', 2);
set(h1, 'Color', 'red');

set(h2, 'LineWidth', 2);
set(h2, 'Color', 'k');
set(h2, 'LineStyle','--');

set(h3, 'LineWidth', 2);
set(h3, 'Color', 'g');
set(h3, 'LineStyle',':');

set(h4, 'LineWidth', 2);
set(h4, 'Color', 'b');
set(h4, 'LineStyle','-.');

%title('SNR','fontsize', 13);
xlabel('INR [dB]','fontsize', 12);
ylabel('CDF','fontsize', 12);
xticks(linspace(-40,10,6));
yticks(linspace(0,1,11))

xline(-6, 'k:', 'LineWidth',1.5);
title ('INR at satellites', fontsize = 12)
ax = gca;
ax.GridLineWidth = 2;
set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));
%legend([h1 h2 h3,h4], 'codebook', 'SVD','tracking based nulling','ideal nulling', ...
%    'Location', 'southeast', 'interpreter', 'latex', 'fontsize', 13);

xlim([-40,10]);
set(gca,'fontname','times new roman');  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nexttile;
%subplot(1,2,1);
 
h1 = cdfplot(snr10_cbook);
hold on;
h2 = cdfplot(snr10_svd);
hold on;

h3 = cdfplot(snr10_cbf_tr);
hold on;
h4 = cdfplot(snr10_cbf_ideal);
hold on;


set(h1, 'LineWidth', 2);
set(h1, 'Color', 'red');

set(h2, 'LineWidth', 2);
set(h2, 'Color', 'k');
set(h2, 'LineStyle','--');

set(h3, 'LineWidth', 2);
set(h3, 'Color', 'g');
set(h3, 'LineStyle',':');

set(h4, 'LineWidth', 2);
set(h4, 'Color', 'b');
set(h4, 'LineStyle','-.');



legend([h1 h2 h3,h4], 'codebook-based nulling', 'SVD','tracking-based nulling','ideal nulling', ...
    'Location', 'southeast', 'interpreter', 'latex', 'fontsize', 13);
grid on;

%set(gca, 'XTick', -10:10:43);
ax = gca;
ax.GridLineWidth = 2;
set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));
title ('SNR at terrestrial UEs', fontsize = 12)

grid on;
lg  = legend('Orientation','Horizontal','NumColumns',2); 
lg.Layout.Tile = 'South';
xlim([10,55]);
yticks(linspace(0,1,11))
lg  = legend('Orientation','Horizontal','NumColumns',2); 
lg.Layout.Tile = 'South';

xlabel('SNR [dB]','fontsize', 12);
set(gca,'fontname','times new roman');  
ylabel('');
exportgraphics(gcf,'figures/inr_snr_cdf.png', 'Resolution',500);