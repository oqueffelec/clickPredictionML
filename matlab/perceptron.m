%% Mon perceptron

clear all

% l le nombre de donn?es, n la dimension des donn?es
l=4;
n=2;

% Systeme OR
x1=[0 0];
x2=[0 1];
x3=[1 0];
x4=[1 1];
x=[x1;x2;x3;x4];
y=[-1 1 1 1];

% Algo perceptron
w1=zeros(n,1);
k=0;
for i=1:l
    if(y(i)*(x(i,:)*w1)<=0)
        w1=w1+y(i)*x(i,:)';
        k=k+1;
    end
end

% HyperPlan associ? 
figure(1)
hold on
title('Systeme OR')
xlabel('x1')
ylabel('x2')
fplot(@(t) -w1(1)/w1(2)*t);
for j=1:l
    if(y(j)==1)
        plot(x(j,1),x(j,2),'b+');
    else
        plot(x(j,1),x(j,2),'r*');
    end
end
hold off
grid on 

% Systeme AND
x1=[0 0];
x4=[0 1];
x3=[1 0];
x2=[1 1];
x=[x1;x2;x3;x4];
y=[-1 -1 -1 1];

% Algo perceptron
w2=zeros(n,1);
k=0;
for i=1:l
    if(y(i)*(x(i,:)*w2)<=0)
        w2=w2+y(i)*x(i,:)';
        k=k+1;
    end
end

% HyperPlan associ? 
figure(2)
fplot(@(z) -w2(1)/w2(2)*z);
hold on
title('Systeme AND')
axis([-5 5 -5 5]);
xlabel('x1')
ylabel('x2')
for j=1:l
    if(y(j)==1)
        plot(x(j,1),x(j,2),'b+');
    else
        plot(x(j,1),x(j,2),'r*');
    end
end
hold off
grid on 

