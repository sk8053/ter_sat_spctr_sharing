clear;

snr10_cbook = importdata('data/SNR_list_12GHz_codebook_10.txt');
snr10_svd = importdata('data/SNR_list_12GHz_SVD_10.txt');
snr10_cbf_ideal = importdata('data/SNR_list_12GHz_CBF_10_8.txt');
snr10_cbf_tr = importdata('data/SNR_list_12GHz_CBF2_10_8.txt');

set(gca,'fontname','times new roman');  

%tiledlayout(1,2)
%subplot(1,2,1);
nexttile;
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


ylabel('');
legend([h1 h2 h3,h4], 'codebook', 'SVD','tracking based nulling','ideal nulling', ...
    'Location', 'southeast', 'interpreter', 'latex', 'fontsize', 13);
grid on;

%set(gca, 'XTick', -10:10:43);
ax = gca;
ax.GridLineWidth = 2;
set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));
title('');
grid on;
lg  = legend('Orientation','Horizontal','NumColumns',2); 
lg.Layout.Tile = 'South';
xlim([10,55]);
yticks(linspace(0,1,11))
lg  = legend('Orientation','Horizontal','NumColumns',2); 
lg.Layout.Tile = 'South';
xlabel('SNR [dB]','fontsize', 12);
ylabel('CDF','fontsize', 12);
%exportgraphics(gcf,'figures/snr_inr_sat_10BS.pdf','Resolution',400, 'ContentType', 'vector')
exportgraphics(gcf,'figures/snr_10BS.png','Resolution',400)

%saveas(gcf,'inr_cdf.pdf')
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf