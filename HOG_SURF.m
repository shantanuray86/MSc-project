function [re, tmp] = HOG_SURF(Img) % The main function starts here
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 if nargin < 3
    filename = fullfile(pwd, 'flat2.jpg');  % Loading the Image
    Img = imread(filename);
   
 end
 %%%%%% Test2 Image HOG Gradient Image %%%%%%

 % %Extract HOG features.
[V5, visualization] = extractHOGFeatures(Img,'CellSize',[30 30]); % Change bin [ x y] size

figure('Name', 'HOG Feature of Test2 Image')
plot(visualization); % Plotting the HOG feature
 
% Initialization
s = 1000;
p1 = 150;  %%% HOG Feature extraction Bin Size1
p2 = 150;  %%% HOG Feature extraction Bin Size2
sigma1 = 10; %% Standard Deviation1 
sigma2 = 16; %% Standard Deviation2
% Loading image
I1 = Img;

if ndims(I1) == 3
    I = rgb2gray(I1); %% Converting Colour Image to Grey Scale
end
array = size(I1);     %% Loading the size of Image
centre1 = floor(array/2);  
for x= 1:array(1)
    for y=1:array(2)
        I(x,y) = 5*I1(x,y);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DOG filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for x=1:array(1)
    for y=1:array(2)
        z = 5;
        G1(x,y) = z*exp(-((x-centre1(1))^2 + (y-centre1(2))^2)/(2*(sigma1^2)));
        
    end
end
for x=1:array(1)
    for y=1:array(2)
        G2(x,y) = z*exp(-((x-centre1(1))^2 + (y-centre1(2))^2)/(2*(sigma2^2)));
    end
end


G = G2-G1;

%%%%%%%%%%Fourier Transform%%%%%%%%%%%%%%%%%%%%%%%

Im(1:512, 1:512,s) = (I);

F1 = fftshift(Im(1:512,1:512,s));
F2 = fft2(F1);
F3 = fftshift (F2);


 
 
%# Initializations:

scale = [0.5  2];              %# The resolution scale factors: [rows columns]
oldSize = size(G);                   %# Get the size of your image
newSize = max(floor(scale.*oldSize(0.5 :2)),2);  %# Compute the new image size

%# Compute an upsampled set of indices:

rowIndex = min(round(((0.5:newSize(1))-0.5)./scale(1)+0.5),oldSize(1));
colIndex = min(round(((0.5:newSize(2))-0.5)./scale(2)+0.5),oldSize(2));

%# Index old image to get new image:

outputImage = G(rowIndex,colIndex,:);

Q= imrotate(outputImage,0,'bilinear','crop');  %% Rotating the Filter Kernel

figure('Name', 'Kernel Angle 0 degree');
imshow(Q);

q = imresize(Q,[512  512]);
w= q.*F3;

%ifftshift%%%%%%%%%%%%%%%%%%%
R2 = ifftshift(w);
R3 = ifft2(R2);
R4 = ifftshift (R3);
R7 = real(R4);
Fig = uint8(R7);

figure('Name', 'Filtered Image at 0 degree');
imshow(Fig);
%%%%% Intensity Histogram of Filtered Image%%%%%%
figure ('Name', ' Histogram of Image')
imhist (Fig);
xlabel('Number of Bins')
ylabel('Intensity Level')

% %Extract HOG features.
[V5, visualization] = extractHOGFeatures(Fig,'CellSize',[p1 p2]);

figure('Name', 'HOG Feature  at 0 degree kernel')
plot(visualization);

%%%%%30 DEGREES %%%%%%%%%
Q1= imrotate(outputImage,30,'nearest','crop');

q1 = imresize(Q1,[512  512]);
figure('Name', 'Kernel Angle 30 degree');
imshow(q1);
w1= q1.*F3;

%ifftshift%%%%%%%%%%%%%%%%%%%
R21 = ifftshift(w1);
R31 = ifft2(R21);
R41 = ifftshift (R31);
R71 = real(R41);


Fig1 = uint8(R71);

figure('Name', 'Filtered Image at 30 degree');
imshow(Fig1);
%%%%% Intensity Histogram of Filtered Image%%%%%%
figure ('Name', ' Histogram of Image')
imhist (Fig1);
xlabel('Number of Bins')
ylabel('Intensity Level')
% %Extract HOG features.
[V5, visualization] = extractHOGFeatures(Fig1,'CellSize',[p1 p2]);

figure('Name', 'HOG Feature  at 30 degree kernel')
plot(visualization);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% 45 DEGREES  %%%%%%%%

Q11= imrotate(outputImage,45,'bilinear','crop');

q11 = imresize(Q11,[512  512]);
figure('Name', 'Kernel Angle 45 degree');
imshow(q11);
w11= q11.*F3;

%ifftshift%%%%%%%%%%%%%%%%%%%
R211 = ifftshift(w11);
R311 = ifft2(R211);
R411 = ifftshift (R311);
R711 = real(R411);


Fig11 = uint8(R711);

figure('Name', 'Filtered Image at 45 degree');
imshow(Fig11);
%%%%% Intensity Histogram of Filtered Image%%%%%%
figure ('Name', ' Histogram of Image')
imhist (Fig11);
xlabel('Number of Bins')
ylabel('Intensity Level')
% %Extract HOG features.
[V5, visualization] = extractHOGFeatures(Fig11,'CellSize',[p1 p2]);

figure('Name', 'HOG Feature at 45 degree kernel')
plot(visualization);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%  90 DEGREES  %%%%%%%%

Q111= imrotate(outputImage,90,'bilinear','crop');


q111 = imresize(Q111,[512  512]);
figure('Name', 'Kernel Angle 90 degree');
imshow(q111);
w111= q111.*F3;

%ifftshift%%%%%%%%%%%%%%%%%%%
R2111 = ifftshift(w111);
R3111 = ifft2(R2111);
R4111 = ifftshift (R3111);
R7111 = real(R4111);


Fig111 = uint8(R7111);

figure('Name', 'Filtered Image at 90 degree');
imshow(Fig111);
%%%%% Intensity Histogram of Filtered Image%%%%%%
figure ('Name', ' Histogram of Image')
imhist (Fig111);
xlabel('Number of Bins')
ylabel('Intensity Level')

% %Extract HOG features.
[V5, visualization] = extractHOGFeatures(Fig111,'CellSize',[p1 p2]);

figure('Name', 'HOG Feature at 90 degree kernel')
plot(visualization);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   135 DEGREES  %%%%%%

Q1111= imrotate(outputImage,135,'bilinear','crop');
q1111 = imresize(Q1111,[512  512]);
figure('Name', 'Kernel Angle 135 degree');
imshow(q1111);
w1111= q1111.*F3;

%ifftshift%%%%%%%%%%%%%%%%%%%
R21111 = ifftshift(w1111);
R31111 = ifft2(R21111);
R41111 = ifftshift (R31111);
R71111 = real(R41111);

Fig1111 = uint8(R71111);

figure('Name', 'Filtered Image at 135 degree');
imshow(Fig1111);
%%%%% Intensity Histogram of Filtered Image%%%%%%
figure ('Name', ' Histogram of Image')
imhist (Fig1111);
xlabel('Number of Bins')
ylabel('Intensity Level')

% %Extract HOG features.
[V5, visualization] = extractHOGFeatures(Fig1111,'CellSize',[p1 p2]);

figure('Name', 'HOG Feature at 135 degree kernel')
plot(visualization);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%  225 DEGREES %%%%%%%

Q11111= imrotate(outputImage,225,'bilinear','crop');

q11111 = imresize(Q11111,[512  512]);
figure('Name', 'Kernel Angle 225 degree');
imshow(q11111);
w11111= q11111.*F3;

%ifftshift%%%%%%%%%%%%%%%%%%%
R211111 = ifftshift(w11111);
R311111 = ifft2(R211111);
R411111 = ifftshift (R311111);
R711111 = real(R411111);
Fig11111 = uint8(R711111);

figure('Name', 'Filtered Image at 225 degree');
imshow(Fig11111);
%%%%% Intensity Histogram of Filtered Image%%%%%%
figure ('Name', ' Histogram of Image')
imhist (Fig11111);
xlabel('Number of Bins')
ylabel('Intensity Level')

% %Extract HOG features.
[V5, visualization] = extractHOGFeatures(Fig1,'CellSize',[p1  p2]);

figure('Name', 'HOG Feature at 225 degree kernel')
plot(visualization);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  270 DEGREES %%%%%%%%%%
Q111111= imrotate(outputImage,270,'bilinear','crop');
q111111 = imresize(Q111111,[512  512]);
figure('Name', 'Kernel Angle 270 degree');
imshow(q111111);
w111111= q111111.*F3;

%ifftshift%%%%%%%%%%%%%%%%%%%
R2111111 = ifftshift(w111111);
R3111111 = ifft2(R2111111);
R4111111 = ifftshift (R3111111);
R7111111 = real(R4111111);
Fig111111 = uint8(R7111111);

figure('Name', 'Filtered Image at 270 degree');
imshow(Fig111111);
%%%%% Intensity Histogram of Filtered Image%%%%%%
figure ('Name', ' Histogram of Image')
imhist (Fig111111);
xlabel('Number of Bins')
ylabel('Intensity Level')
% %Extract HOG features.
[V5, visualization] = extractHOGFeatures(Fig111111,'CellSize',[p1  p2]);

figure('Name', 'HOG Feature at 270 degree kernel')
plot(visualization);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  315 DEGREES %%%%%%%%%%
Q1111111= imrotate(outputImage,315,'bilinear','crop');
q1111111 = imresize(Q1111111,[512  512]);
figure('Name', 'Kernel Angle 315 degree');
imshow(q1111111);
w1111111= q1111111.*F3;

%ifftshift%%%%%%%%%%%%%%%%%%%
R21111111 = ifftshift(w1111111);
R31111111 = ifft2(R21111111);
R41111111 = ifftshift (R31111111);
R71111111 = real(R41111111);
Fig1111111 = uint8(R71111111);
figure('Name', 'Filtered Image at 315 degree');
imshow(Fig1111111);
%%%%% Intensity Histogram of Filtered Image%%%%%%
figure ('Name', ' Histogram of Image')
imhist (Fig1111111);
xlabel('Number of Bins')
ylabel('Intensity Level')
%% %Extract HOG features %%%%%%%%%%%%%
[V5, visualization] = extractHOGFeatures(Fig1111111,'CellSize',[p1 p2]);

figure('Name', 'HOG Feature at 315 degree kernel')
plot(visualization);

%%%%%%%%Plotting the HISTOGRAM at Different KERNEL Angles %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[counts1, binCenters1] = hist(Fig(:), 0:5:460);
[counts2, binCenters2] = hist(Fig1(:), 30:5:460);
[counts3, binCenters3] = hist(Fig11(:), 45:5:460);
[counts4, binCenters4] = hist(Fig111(:), 90:5:460);
[counts5, binCenters5] = hist(Fig1111(:), 135:5:460);
[counts6, binCenters6] = hist(Fig11111(:), 225:5:460);
[counts7, binCenters7] = hist(Fig111111(:), 270:5:460);
[counts8, binCenters8] = hist(Fig1111111(:), 315:5:460);
figure;

bar(binCenters1, counts1,'facecolor','r' );
 hold on;
%plot(binCenters2, counts2, 'g-');
bar(binCenters2, counts2,'facecolor','g' );
hold on;
%plot(binCenters3, counts3, 'b-');
bar(binCenters3, counts3,'facecolor','b' );
bar(binCenters4, counts4,'facecolor','y' );
bar(binCenters5, counts5,'facecolor','c' );
bar(binCenters6, counts6,'facecolor','k' );
bar(binCenters7, counts7,'facecolor','m');
bar(binCenters8, counts8,'facecolor','w' );
grid on;

%%%%%%%%%%%%%% HISTOGRAM IDENTIFICATION %%%%%%%%%%%%%%%%
I1 = imread('lena.png');
Noise = 0.05;
J = imnoise(I1,'Gaussian',0,Noise); % Adding Gaussian Noise %%% 

% Calculate the Normalized Histogram of Image 1 and Image 2
hn1 = imhist(I1)./numel(I1);
hn2 = imhist(J)./numel(J);
 %%%% PLotting the Histograms %%%%%%%
subplot(2,2,1);subimage(I1)
subplot(2,2,2);subimage(J)
subplot(2,2,3);plot(hn1)
xlabel('Number of Bins')
ylabel('Intensity Level')
subplot(2,2,4);plot(hn2)
xlabel('Number of Bins')
ylabel('Intensity Level')
% Calculate the histogram error %%%%%%
f10 = sum((hn1 - hn2).^2);
disp(f10) %display the result to console %%%%%%

% Displaying the Result of Histogram Identification%%

if  (f10>0)&& (f10<0.05)
    disp('Histogram match.') 
else
    disp('Histogram does not match.')
end

 %%%%%%%%%%%%  SURF   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = -180; %%% Angle of Rotation of Image%%%%%
I2 = imresize(imrotate(I1,A), 1.2);
 
points1 = detectSURFFeatures(I1);  %%% SURF Feature Detection    %%%%%%
points2 = detectSURFFeatures(I2);  %%% SURF Feature Detection    %%%%%%
[f1, vpts1] = extractFeatures(I1, points1); %%% SURF Feature Extraction %%%%%%
[f2, vpts2] = extractFeatures(I2, points2); %%% SURF Feature Extraction  %%%%%%
indexPairs = matchFeatures(f1, f2) ; %%% SURF Feature Points Matching  %%%%%%
matchedPoints1 = vpts1(indexPairs(:, 1));
matchedPoints2 = vpts2(indexPairs(:, 2));
figure('Name', 'SURF Features');
ax = axes;
showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2,'montage'); % Displaying the Matched Features  %%
title(ax, 'Putative point matches');
legend(ax,'Matched points 1','Matched points 2');

