%% Stochastic Gradient Descent

clear all

% Soit xj une instance d'une pub avec ses differents features et yj le
% label 

x1=[4 5 6 3 5 6 3 5 6 2]';
x2=[4 5 6 3 5 6 3 5 6 2]';
x3=[4 5 6 3 5 6 3 5 6 2]';
x4=[4 5 6 3 5 6 3 5 6 2]';
x5=[3 9 6 3 5 6 3 5 6 2]';
x6=[2 5 29 3 5 6 3 5 6 2]';
x7=[1 5 6 32 5 6 3 5 6 2]';
x8=[7 5 6 3 45 54 3 5 6 2]';
x9=[5 5 6 3 5 22 3 34 6 2]';
x10=[0 5 6 3 5 6 35 5 76 2]';

x=[x1 x2 x3 x4 x5 x6 x7 x8 x9 x10];
y=[0 0 1 0 1 1 1 1 1 0];

% Determinons les w en appliquant une methode iterative
    n=0.000001;
    w_old=ones(10,1);
    

for j=1:10
    h=(y(j)-x(:,j)'*w_old)*x(:,j);
    w_new=w_old-h*n; 
    w_old=w_new;
end

% Creation du vecteur y_chap  

y_chap1=x(:,1)'*w_new;
y_chap2=x(:,2)'*w_new;
y_chap3=x(:,3)'*w_new;
y_chap4=x(:,4)'*w_new;
y_chap5=x(:,5)'*w_new;
y_chap6=x(:,6)'*w_new;
y_chap7=x(:,7)'*w_new;
y_chap8=x(:,8)'*w_new;
y_chap9=x(:,9)'*w_new;
y_chap10=x(:,10)'*w_new;

y_chap=[y_chap1 y_chap2 y_chap3 y_chap4 y_chap5 y_chap6 y_chap7 y_chap8 y_chap9 y_chap10];

% Implementation average loss L 

L=zeros(1,8);
T=[1,2,3,4,5, 6, 9,10];
for i=1:length(T)
    for j=1:T(i)
        L(i)=L(i)+(y_chap(j)-y(j))^2;
    end
    L(i)=L(i)/T(i);
end


figure(1)
hold on
title('Average Loss')
xlabel('nbre instances')
plot(L)

