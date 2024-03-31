function [re, tmp] = FirstStep(Img)% The function begins here with filename
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reading the Image
 if nargin < 3
    filename = fullfile(pwd, 'lenac.jpg'); %The image is loaded under Filename
    Img = imread(filename);

 end
% Initialization
s = 10;        % Scale of the image
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
figure('Name','FFTshift Image');
imshow(F1);

w= G.*F3;   % Multiplying the DOG with FT image
figure(2);
imshow(F2);
%ifftshift%%%%%%%%%%%%%%%%%%%
R2 = ifftshift(w);
R3 = ifft2(R2);
R4 = ifftshift (R3); % Taking inverse fourier transform
R7 = real(R4);       % Showing the real part of the Image
R17 = uint8(R7);

figure('Name','Filtered image');
imshow(R17);
figure (10);
imshow(Img);
figure (11);
imshow (I1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ADDITION OF GAUSSIAN NOISE %%%%%%%%%%%%%%%
J = imnoise(Img,'Gaussian',0.05); % adding Gaussian Noise 
d = rgb2gray(J);
figure('Name','Gaussian Noise Distorted Image');


imshow(J);
figure('Name','Gaussian Noise Distorted Image grey scale');
imshow(d);
%%% Filtering Noise Distorted Image with DOG%%%%%%
Im1(1:512, 1:512,s) = (d);

F11 = fftshift(Im1(1:512,1:512,s));
F21 = fft2(F11);
F31 = fftshift (F21);


w1= G.*F31;   % Multiplying the DOG with FT image

%ifftshift%%%%%%%%%%%%%%%%%%%
R21 = ifftshift(w1);
R31 = ifft2(R21);
R41 = ifftshift (R31); % Taking inverse fourier transform
R71 = real(R41);       % Showing the real part of the Image
R711 = uint8(R71);


figure('Name','Filtered Noise Distorted Image');
imshow(R711);
%%%%%%%%%%%%%% Standard Deviation Curve %%%%%%%%%%%%
gauss_size = size(G);
Gx = G(1:gauss_size(1),ceil(gauss_size(2)/2));
Gx = flipud(Gx);
z = 0:1/(gauss_size(1)-1):1;
figure(90909);
plot (z,Gx);
xlabel('Standard Deviation')
ylabel('Manitude')

figure('Name', 'Filter Kernel Size');
imshow(G);







