clear;
inr10_CBF = importdata('data8t8/INR_list_12GHz_CBF_10_2.txt');
inr10_svd = importdata('data8t8/INR_list_12GHz_SVD_10.txt');
inr10_cbook = importdata('data8t8/INR_list_12GHz_codebook_10.txt');
%inr10_cbf_ideal = importdata('data8t8/downlink_inr_12GHz_50_CBF2_3.txt');
%inr10_cbf_tr = importdata('data8t8/downlink_inr_12GHz_10_CBF2_8.txt');

%snr10_cbook = importdata('SNR_list_12GHz_codebook_10.txt');
%snr10_svd = importdata('SNR_list_12GHz_SVD_10.txt');
%snr10_cbf_8 = importdata('SNR_list_12GHz_CBF_10_8.txt');
%snr10_cbf_9 = importdata('SNR_list_12GHz_CBF_10_9.txt');

set(gca,'fontname','times new roman');  

%tiledlayout(1,2)
%subplot(1,2,1);
nexttile;
h1 = cdfplot(inr10_svd);
hold on;
h2 = cdfplot(inr10_CBF);
hold on;

h3 = cdfplot(inr10_cbook);
hold on;
%h4 = cdfplot(inr10_cbf_ideal);
%hold on;


set(h1, 'LineWidth', 2);
set(h1, 'Color', 'red');

set(h2, 'LineWidth', 2);
set(h2, 'Color', 'k');
set(h2, 'LineStyle','--');

set(h3, 'LineWidth', 2);
set(h3, 'Color', 'g');
set(h3, 'LineStyle',':');

%set(h4, 'LineWidth', 2);
%set(h4, 'Color', 'b');
%set(h4, 'LineStyle','-.');

%title('SNR','fontsize', 13);
xlabel('INR [dB]','fontsize', 12);
ylabel('CDF','fontsize', 12);
%xline(-6, 'k:', 'LineWidth',1.5);
legend( 'SVD','tracking-based nulling $\lambda = 10^2$','codebook', ...
    'Location', 'northwest', 'interpreter', 'latex', 'fontsize', 13);


%legend([h1 h2], '6 GHz', '18 GHz', 'Location', 'northwest');

%xlim([-80,40]);
%xticks(linspace(-40,10,6));
yticks(linspace(0,1,11))
%set(gca, 'XTick', -10:10:43);
ax = gca;
ax.GridLineWidth = 2;
%set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));
title('');
grid on;
%lg  = legend('Orientation','Horizontal','NumColumns'); 
%lg.Layout.Tile = 'South';

%exportgraphics(gcf,'figures/snr_inr_sat_10BS.pdf','Resolution',400, 'ContentType', 'vector')
exportgraphics(gcf,'figures/terr_inr_10BS.png','Resolution',600)

%saveas(gcf,'inr_cdf.pdf')
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf