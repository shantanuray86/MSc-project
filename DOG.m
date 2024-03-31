function [re, tmp] = DOG(Img)% The function begins here with filename
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reading the Image
 if nargin < 3
    filename = fullfile(pwd, 'lena.png'); %The image is loaded under Filename
    Img = imread(filename);

 end
% Initialization
s = 1000;        % Scale of the image
sigma1 = 10 ; % Standard Deviation1
sigma2 = 16;  % Standard Deviation2
% Loading image
I1 = Img;
if ndims(I1) == 3
    I1 = rgb2gray(I1); % Colour to Grey Scale Conversion
end
array = size(I1);      % Reading the size of the image
centre1 = floor(array/2);
for x= 1:array(1)
    for y=1:array(2)
        I(x,y) = 1*I1(x,y);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 Gaussian filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z = 5;
for x=1:array(1)
    for y=1:array(2)
        G1(x,y) = z*exp(-((x-centre1(1))^2 + (y-centre1(2))^2)/(2*(sigma1^2)));
        
    end  %  The Second Order Derivative of Gaussian (First one)
end
for x=1:array(1)
    for y=1:array(2)
        G2(x,y) = z*exp(-((x-centre1(1))^2 + (y-centre1(2))^2)/(2*(sigma2^2)));
    end
end      %  The Second Order Derivative of Gaussian (Second  one)


G = G2-G1;  % Difference of Gaussian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%Fourier Transform%%%%%%%%%%%%%%%%%%%%%%%

Im(1:512, 1:512,s) = (I);

F1 = fftshift(Im(1:512,1:512,s));
F2 = fft2(F1);
F3 = fftshift (F2);


scale = [0.7  2];              %# The resolution scale factors: [rows columns]
oldSize = size(G);                   %# Get the size of your image
newSize = max(floor(scale.*oldSize(0.5 :2)),2);  %# Compute the new image size

%# Compute an upsampled set of indices:

rowIndex = min(round(((0.5:newSize(1))-0.5)./scale(1)+0.5),oldSize(1));
colIndex = min(round(((0.5:newSize(2))-0.5)./scale(2)+0.5),oldSize(2));

%# Index old image to get new image:

outputImage = G(rowIndex,colIndex,:); % Forming the new scaled image

j= 90;  % Kernel angle 

Q= imrotate(outputImage,j,'bilinear','crop'); % Rotating the Kernel

l= 45;  % Kernel angle
Q1= imrotate(outputImage,l,'bilinear','crop'); % Rotating the Kernel

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q = imresize(Q,[512  512]);
figure('Name','Rotated Kernel at j angle');
imshow(q);

w= q.*F3;  % Multiplying the filter kernel with fft2 image
figure('Name','Phase information');
imshow(F2);
%ifftshift%%%%%%%%%%%%%%%%%%% Inverse Fourier Transform%%%%%%%%Case 1 %%%
R2 = ifftshift(w); 
R3 = ifft2(R2);
R4 = ifftshift (R3);
R7 = real(R4);
R8 = uint8(R7);

figure('Name','Filtered Image');
imshow(R8);

figure ('Name','Original Grey Scale Image');
imshow (I1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Case 2  %%%
% Initialization
s = 1000;        % Scale of the image
sigma1 = 10 ; % Standard Deviation1
sigma2 = 16;  % Standard Deviation2
% Loading image
I1 = Img;
if ndims(I1) == 3
    I1 = rgb2gray(I1); % Colour to Grey Scale Conversion
end
array = size(I1);      % Reading the size of the image
centre1 = floor(array/2);
for x= 1:array(1)
    for y=1:array(2)
        I(x,y) = 1*I1(x,y);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 Gaussian filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z1 = 5;
for x=1:array(1)
    for y=1:array(2)
        G11(x,y) = z1*exp(-((x-centre1(1))^2 + (y-centre1(2))^2)/(2*(sigma1^2)));
        
    end  %  The Second Order Derivative of Gaussian (First one)
end
for x=1:array(1)
    for y=1:array(2)
        G21(x,y) = z1*exp(-((x-centre1(1))^2 + (y-centre1(2))^2)/(2*(sigma2^2)));
    end
end      %  The Second Order Derivative of Gaussian (Second  one)


G1 = G21-G11;  % Difference of Gaussian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%Fourier Transform%%%%%%%%%%%%%%%%%%%%%%%

Im(1:512, 1:512,s) = (I);

F1 = fftshift(Im(1:512,1:512,s));
F2 = fft2(F1);
F3 = fftshift (F2);


scale = [0.2  2];              %# The resolution scale factors: [rows columns]
oldSize = size(G1);                   %# Get the size of your image
newSize = max(floor(scale.*oldSize(0.5 :2)),2);  %# Compute the new image size

%# Compute an upsampled set of indices:

rowIndex = min(round(((0.5:newSize(1))-0.5)./scale(1)+0.5),oldSize(1));
colIndex = min(round(((0.5:newSize(2))-0.5)./scale(2)+0.5),oldSize(2));

%# Index old image to get new image:

outputImage1 = G1(rowIndex,colIndex,:); % Forming the new scaled image
l= 45;  % Kernel angle
Q1= imrotate(outputImage1,l,'bilinear','crop'); % Rotating the Kernel
q1 = imresize(Q1,[512  512]);
figure('Name','Rotated Kernel at l angle case 2');
imshow(q1);

w1= q1.*F3;  % Multiplying the filter kernel with fft2 image
figure('Name','Phase information case 2');
imshow(F2);
%ifftshift%%%%%%%%%%%%%%%%%%% Inverse Fourier Transform%%%%%%%%
R21 = ifftshift(w1); 
R31 = ifft2(R21);
R41 = ifftshift (R31);
R71 = real(R41);
R81 = uint8(R71);

figure('Name','Filtered Image case 2');
imshow(R81);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Case 3  %%%
% Initialization
s = 1000;        % Scale of the image
sigma1 = 50 ; % Standard Deviation1
sigma2 = 80;  % Standard Deviation2
% Loading image
I1 = Img;
if ndims(I1) == 3
    I1 = rgb2gray(I1); % Colour to Grey Scale Conversion
end
array = size(I1);      % Reading the size of the image
centre1 = floor(array/2);
for x= 1:array(1)
    for y=1:array(2)
        I(x,y) = 1*I1(x,y);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 Gaussian filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z1 = 5;
for x=1:array(1)
    for y=1:array(2)
        G111(x,y) = z1*exp(-((x-centre1(1))^2 + (y-centre1(2))^2)/(2*(sigma1^2)));
        
    end  %  The Second Order Derivative of Gaussian (First one)
end
for x=1:array(1)
    for y=1:array(2)
        G211(x,y) = z1*exp(-((x-centre1(1))^2 + (y-centre1(2))^2)/(2*(sigma2^2)));
    end
end      %  The Second Order Derivative of Gaussian (Second  one)


G2 = G211-G111;  % Difference of Gaussian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%Fourier Transform%%%%%%%%%%%%%%%%%%%%%%%

Im(1:512, 1:512,s) = (I);

F111 = fftshift(Im(1:512,1:512,s));
F211 = fft2(F111);
F311 = fftshift (F211);


scale = [0.7  2];              %# The resolution scale factors: [rows columns]
oldSize = size(G2);                   %# Get the size of your image
newSize = max(floor(scale.*oldSize(0.5 :2)),2);  %# Compute the new image size

%# Compute an upsampled set of indices:

rowIndex = min(round(((0.5:newSize(1))-0.5)./scale(1)+0.5),oldSize(1));
colIndex = min(round(((0.5:newSize(2))-0.5)./scale(2)+0.5),oldSize(2));

%# Index old image to get new image:

outputImage11 = G2(rowIndex,colIndex,:); % Forming the new scaled image
t= 135;  % Kernel angle
Q11= imrotate(outputImage11,t,'bilinear','crop'); % Rotating the Kernel
q11 = imresize(Q11,[512  512]);
figure('Name','Rotated Kernel at t angle case 3');
imshow(q11);

w11= q11.*F311;  % Multiplying the filter kernel with fft2 image
figure('Name','Phase information case 3');
imshow(F211);
%ifftshift%%%%%%%%%%%%%%%%%%% Inverse Fourier Transform%%%%%%%%
R211 = ifftshift(w11); 
R311 = ifft2(R211);
R411 = ifftshift (R311);
R711 = real(R411);
R811 = uint8(R711);

figure('Name','Filtered Image case 3');
imshow(R811);
