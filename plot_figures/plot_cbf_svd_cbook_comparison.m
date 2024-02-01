clear;
inr10_cbook = importdata('downlink_inr_12GHz_10_codebook.txt');
inr10_svd = importdata('downlink_inr_12GHz_10_SVD.txt');
inr10_cbf_8 = importdata('downlink_inr_12GHz_10_CBF_8.txt');
inr10_cbf_9 = importdata('downlink_inr_12GHz_10_CBF_9.txt');

snr10_cbook = importdata('SNR_list_12GHz_codebook_10.txt');
snr10_svd = importdata('SNR_list_12GHz_SVD_10.txt');
snr10_cbf_8 = importdata('SNR_list_12GHz_CBF_10_8.txt');
snr10_cbf_9 = importdata('SNR_list_12GHz_CBF_10_9.txt');

set(gca,'fontname','times new roman');  

tiledlayout(1,2)
%subplot(1,2,1);
nexttile;
h1 = cdfplot(snr10_cbook);
hold on;
h2 = cdfplot(snr10_svd);
hold on;

h3 = cdfplot(snr10_cbf_8);
hold on;
h4 = cdfplot(snr10_cbf_9);
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

%title('SNR','fontsize', 13);
xlabel('SNR [dB]','fontsize', 11);
ylabel('');
%legend([h1 h2 h3,h4], 'codebook', 'SVD','online-nulling, $\lambda=10^8$','online-nulling, $\lambda=10^9$', ...
%    'Location', 'southeast', 'interpreter', 'latex', 'fontsize', 11);
grid on;
xlim([10,53]);
set(gca, 'XTick', -10:10:43);
title('');


%subplot(1,2,2);
nexttile;
h1 = cdfplot(inr10_cbook);
hold on;
h2 = cdfplot(inr10_svd);
hold on;

h3 = cdfplot(inr10_cbf_8);
hold on;

h4 = cdfplot(inr10_cbf_9);
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

%title('SINR', 'fontsize', 13);
xlabel('INR [dB]','fontsize', 11);
ylabel('');

legend([h1 h2 h3,h4], 'codebook', 'SVD','online-nulling, $\lambda=10^8$','online-nulling, $\lambda=10^9$', ...
    'Location', 'southeast', 'interpreter', 'latex', 'fontsize', 11);
grid on;
xlim([-50,25]);
%set(gca, 'XTick', -10:10:43);

title('');
sgtitle('SNR of terrestrial UEs and INR of satellites (N$_{BS}$ = 10)', 'fontsize', 12, 'interpreter', 'latex')
% add a bit space to the figure
lg  = legend('Orientation','Horizontal','NumColumns',2); 
lg.Layout.Tile = 'South';

%exportgraphics(gcf,'figures/snr_inr_sat_10BS.pdf','Resolution',400, 'ContentType', 'vector')
exportgraphics(gcf,'figures/snr_inr_sat_10BS.png','Resolution',400)

%saveas(gcf,'inr_cdf.pdf')
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf