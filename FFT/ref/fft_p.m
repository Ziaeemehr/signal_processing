clc;
clear all
close all;
N=2^10;
dt=1./800; %s
T=N.*dt;
t=linspace(0,T,N);
f=linspace(-1./(2*dt),1./(2*dt),N);
nu1=50;
nu2=110;
%y=3.*sin(2*pi*nu1.*t)+6.*cos(2*pi*nu2.*t);
y=3.*exp(2.*1i.*pi*nu1.*t)+6.*exp(2.*1i*pi*-nu2.*t);
figure
plot(t,real(y));
fy=fft(y).*1./N;
fy=fftshift(fy);
figure
plot(f,abs(fy))