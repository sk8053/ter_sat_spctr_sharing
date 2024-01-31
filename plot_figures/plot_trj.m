clear;

M2 = csvread('obs.csv');
lat_o = M2(:,2);
lon_o = M2(:,1);
h_o = ones(length(lat_o),1)*5000;

filePattern = fullfile('STARLINK-*.csv');
files = dir(filePattern);
uif = uifigure;
g = geoglobe(uif, 'NextPlot', 'add');
for i=1:length(files)
    M = csvread(files(i).name);
    lat = M(:,1);
    lon = M(:,2);
    h = ones(length(lat),1)*5000;

    mskip = 1:length(lat);
    %geoplot(lat, lon)
    geoplot3(g,lat,lon,h,"o-","MarkerIndices",mskip);
    hold(g, 'on');
end
geoplot3(g,lat_o,lon_o,h_o,"ko", "Markersize", 12);
hold off;