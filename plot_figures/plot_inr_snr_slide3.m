clear;
inr10_c = importdata('downlink_inr_18GHz_10_codebook.txt');
snr10_c = importdata('SNR_list_18GHz_codebook_10.txt');
sinr10_c = importdata('SINR_list_18GHz_codebook_10.txt');

inr10_s = importdata('downlink_inr_18GHz_10_SVD.txt');
snr10_s = importdata('SNR_list_18GHz_SVD_10.txt');
sinr10_s = importdata('SINR_list_18GHz_SVD_10.txt');

set(gca,'fontname','times new roman');  
%subplot(1,2,1);
figure(1);
h1 = cdfplot(inr10_c);
hold on;
h2 = cdfplot(inr10_s);
hold on;

set(h1, 'LineWidth', 2);
set(h1, 'Color', 'red');
set(h2, 'LineWidth', 2);
set(h2, 'Color', 'k');
set(h2, 'LineStyle','--');


title('INR of satellites');
xlabel('INR [dB]');
ylabel('CDF');
legend('codebook','SVD', 'Location', 'northwest', 'interpreter', 'latex');
grid on;
xlim([-40,22]);
exportgraphics(gcf,'figures/inr_sat_10BS.png','Resolution',400)

subplot(1,2,1);
%figure(2);
h1 = cdfplot(snr10_c);
hold on;
h2 = cdfplot(snr10_s);
hold on;

set(h1, 'LineWidth', 2);
set(h1, 'Color', 'red');

set(h2, 'LineWidth', 2);
set(h2, 'Color', 'k');
set(h2, 'LineStyle','--');

%title('SNR','fontsize', 13);
xlabel('SNR [dB]','fontsize', 11);
ylabel('');
legend('codebook', 'SVD', 'Location', 'northwest', 'interpreter', 'latex', 'fontsize', 11);
grid on;
xlim([-10,43]);
set(gca, 'XTick', -10:10:43);

title('');
%exportgraphics(gcf,'figures/snr_sat_10BS.png','Resolution',400)

subplot(1,2,2);
%figure(3);
h1 = cdfplot(sinr10_c);
hold on;
h2 = cdfplot(sinr10_s);
hold on;

set(h1, 'LineWidth', 2);
set(h1, 'Color', 'red');

set(h2, 'LineWidth', 2);
set(h2, 'Color', 'k');
set(h2, 'LineStyle','--');

%title('SINR', 'fontsize', 13);
xlabel('SINR [dB]','fontsize', 11);
ylabel('');
legend('codebook', 'SVD', 'Location', 'northwest', 'interpreter', 'latex', 'fontsize', 11);
grid on;
xlim([-10,43]);
set(gca, 'XTick', -10:10:43);

title('');
sgtitle('SNR and SINR of terrestrial UEs', 'fontsize', 13)
exportgraphics(gcf,'figures/snr_sinr_sat_10BS.png','Resolution',400)

%exportgraphics(gcf,'inr_cdf.png','Resolution',400)
%saveas(gcf,'inr_cdf.pdf')
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf