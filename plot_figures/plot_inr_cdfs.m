dl_inr2 = importdata('downlink_inr_2.txt');
dl_inr6 = importdata('downlink_inr_6.txt');
%dl_inr8 = importdata('downlink_inr_8.txt');
dl_inr10 = importdata('downlink_inr_10.txt');

up_inr2 = importdata('uplink_inr_2.txt');
up_inr6 = importdata('uplink_inr_6.txt');
%up_inr8 = importdata('uplink_inr_8.txt');
up_inr10 = importdata('uplink_inr_10.txt');

set(gca,'fontname','times new roman');  
subplot(1,2,1);
h1 = cdfplot(up_inr2);
hold on;
h2 = cdfplot(up_inr6);
hold on;
h3 = cdfplot(up_inr10);
set(h1, 'LineWidth', 2);
set(h1, 'Color', 'red');

set(h2, 'LineWidth', 2);
set(h2, 'Color', 'k');
set(h2, 'LineStyle','--');

set(h3, 'LineWidth', 2);
set(h3, 'Color', 'b');
set(h3, 'LineStyle',':');

title('INR from Uplink');
xlabel('INR [dB]');
ylabel('CDF');
legend('N$_{UE}=2$', 'N$_{UE}=6$','N$_{UE}=10$', 'Location', 'northwest', 'interpreter', 'latex');
grid on;
xlim([-30,9]);

subplot(1,2,2);
h1 = cdfplot(dl_inr2);
hold on;
h2 = cdfplot(dl_inr6);
hold on;
h3 = cdfplot(dl_inr10);

set(h1, 'LineWidth', 2);
set(h1, 'Color', 'red');

set(h2, 'LineWidth', 2);
set(h2, 'Color', 'k');
set(h2, 'LineStyle','--');

set(h3, 'LineWidth', 2);
set(h3, 'Color', 'b');
set(h3, 'LineStyle',':');

title('INR from Downlink');
xlabel('INR [dB]');
ylabel('');
legend('N$_{BS}=2$', 'N$_{BS}=6$','N$_{BS}=10$', 'Location', 'northwest', 'interpreter', 'latex');
grid on;
xlim([-30,20]);
exportgraphics(gcf,'inr_cdf.png','Resolution',400)
%saveas(gcf,'inr_cdf.pdf')
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf
%print -painters -dpdf -r300 -fillpage inr_cdf.pdf