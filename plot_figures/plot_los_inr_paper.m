clear;
inr_all = importdata('data8t8/downlink_inr_12GHz_SVD_21.txt');
elem_all = importdata('data8t8/downlink_elev_ang_12GHz_SVD_21.txt');


I1 = (elem_all>=25) &(elem_all<=45);
I2 = (elem_all>45) &(elem_all<=70);
I3 = (elem_all>70);
inr1 = inr_all(I1);
inr2 = inr_all(I2);
inr3 = inr_all(I3);

t = tiledlayout(1,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
%subplot(1,2,1);
nexttile;
set(gca,'fontname','times new roman');  
h1 = plot(sort(inr_all), linspace(0,1,length(inr_all)));
set(h1, 'LineWidth', 1.2, 'LineStyle','-');
set(h1, 'Color', 'k');
hold on;

h2 = plot(sort(inr1), linspace(0,1,length(inr1)));
hold on;
set(h2, 'LineWidth', 2.5, 'LineStyle',':');
set(h2, 'Color', 'r');

h3 = plot(sort(inr2), linspace(0,1,length(inr2)));
hold on;
set(h3, 'LineWidth', 2.5, 'LineStyle','-.');
set(h3, 'Color', 'b');

h4 = plot(sort(inr3), linspace(0,1,length(inr3)));
hold on;
set(h4, 'LineWidth', 2.5, 'LineStyle','--');
h4.Color = 	"#A2142F";
%set(h4, 'Color', 'g');


ax = gca;
ax.GridLineWidth = 2;
ax.XAxis.FontSize = 11;
ax.YAxis.FontSize = 11;
xline(-6, 'k:', 'LineWidth',1.5);
set(gca, 'XTick', sort([-6, get(gca, 'XTick')]));
grid on;
legend ('$\theta \geq 25^\circ$', '$45^\circ\geq \theta \geq 25^\circ$', ...
    '$70^\circ\geq \theta > 45^\circ$', ...
    '$\theta>70^\circ$', ...
    'Interpreter','latex','fontsize', 12, 'location','southeast' )
%xticks('fontsize', 12);
yticks(linspace(0,1,11))
xlabel('INR [dB]', 'fontsize', 13, ...
    'Interpreter','latex');
ylabel ('CDF', 'FontSize',13,...
    'Interpreter','latex');
xlim([-15,20])
%title ('INR of satellites', fontsize = 13);
title('')
set(gcf,'Position',[100 100 500 300]);
%exportgraphics(gcf,'figures/los_inrs.png', 'Resolution',400);
%exportgraphics(gcf,'figures/inr_elev.png', 'Resolution',600);
exportgraphics(gcf,'figures/inr_elev.eps');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t = tiledlayout(1,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
%subplot(1,2,1);
nexttile;
set(gca,'fontname','times new roman');
h = histogram(elem_all,17,'Normalization','pdf');
%h.Values = h.Values/60;
grid on;
ax = gca;
ax = gca;
ax.GridLineWidth = 2;
ax.XAxis.FontSize = 11;
ax.YAxis.FontSize = 11;
set(gca, 'XTick', sort([25, get(gca, 'XTick')]));
xlabel('Elevation angle [$^\circ$]','interpreter', 'latex', 'fontsize', 13);
ylabel ('PDF', 'FontSize',13, 'Interpreter','latex', 'fontsize', 13);
%title ('PDF of satellite elevation angles', fontsize =13);

title('');
set(gcf,'Position',[100 100 500 300]);
%exportgraphics(gcf,'figures/elev_pdf.png', 'Resolution',600);
exportgraphics(gcf,'figures/elev_pdf.eps');
