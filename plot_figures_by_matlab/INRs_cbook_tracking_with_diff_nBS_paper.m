clear;
inr10 = importdata('data/downlink_inr_12GHz_10_CBF2_8.txt');
inr20 = importdata('data/downlink_inr_12GHz_20_CBF2_8.txt');
inr50 = importdata('data/downlink_inr_12GHz_50_CBF2_8.txt');

inr10_cb = importdata('data/downlink_inr_12GHz_10_codebook.txt');
inr20_cb = importdata('data/downlink_inr_12GHz_20_codebook.txt');
inr50_cb = importdata('data/downlink_inr_12GHz_50_codebook.txt');

t = tiledlayout(1,2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile;

h1 = cdfplot(inr10);
hold on;
h2 = cdfplot(inr20);
hold on;

%h3 = cdfplot(inr_svd10);
hold on;
h4 = cdfplot(inr50);
hold on;


set(h1, 'LineWidth', 2);
set(h1, 'Color', 'red');

set(h2, 'LineWidth', 2);
set(h2, 'Color', 'k');
set(h2, 'LineStyle','--');

%set(h3, 'LineWidth', 2);
%set(h3, 'Color', 'r');
%set(h3, 'LineStyle',':');

set(h4, 'LineWidth', 2);
set(h4, 'Color', 'b');
set(h4, 'LineStyle','-.');

%title('SNR','fontsize', 13);
xlabel('INR [dB]','fontsize', 13);
ylabel('CDF','fontsize', 13);
xline(-6, 'k:', 'LineWidth',1.5);
grid on;
xlim([-35,8]);
title ('tracking-based nulling', fontsize = 14)
ax = gca;
ax.GridLineWidth = 2;
set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));
set(gca,'fontname','times new roman');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nexttile;
h1 = cdfplot(inr10_cb);
hold on;
h2 = cdfplot(inr20_cb);
hold on;

%h3 = cdfplot(inr_svd10);
hold on;
h4 = cdfplot(inr50_cb);
hold on;


set(h1, 'LineWidth', 2);
set(h1, 'Color', 'red');

set(h2, 'LineWidth', 2);
set(h2, 'Color', 'k');
set(h2, 'LineStyle','--');

%set(h3, 'LineWidth', 2);
%set(h3, 'Color', 'r');
%set(h3, 'LineStyle',':');

set(h4, 'LineWidth', 2);
set(h4, 'Color', 'b');
set(h4, 'LineStyle','-.');

%title('SNR','fontsize', 13);
xlabel('INR [dB]','fontsize', 13);
ylabel('');
legend([h1 h2 h4], 'N$_{\rm BS}$=10', 'N$_{\rm BS}$=20','N$_{\rm  BS}$=50', ...
    'Location', 'southeast', 'interpreter', 'latex', 'fontsize', 11);
grid on;
xline(-6, 'k:', 'LineWidth',1.5);
xlim([-35,8]);
title ('codebook-based nulling', fontsize = 14)

legend([h1 h2 h4], 'N$_{\rm BS}$=10', 'N$_{\rm BS}$=20','N$_{\rm  BS}$=50', ...
    'Location', 'southeast', 'interpreter', 'latex', 'fontsize', 13);
ax = gca;
ax.GridLineWidth = 2;
set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));

lg  = legend('Orientation','Horizontal','NumColumns',3); 
lg.Layout.Tile = 'South';
set(gca,'fontname','times new roman');

%exportgraphics(gcf,'figures/snr_inr_sat_10BS.pdf','Resolution',400, 'ContentType', 'vector')
exportgraphics(gcf,'figures/inr_with_diff_nBS.png','Resolution',400)

%saveas(gcf,'inr_cdf.pdf')
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf