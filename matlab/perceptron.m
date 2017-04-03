%% Mon perceptron

clear all

% l le nombre de donn?es, n la dimension des donn?es
l=4;
n=2;

% Systeme OR
x1=[0 0 1];
x2=[0 1 1];
x3=[1 0 1];
x4=[1 1 1];
x=[x1;x2;x3;x4];
y=[-1 1 1 1];

% Algo perceptron
w1=zeros(n+1,1);
k=0;
K=67;
for compteur=1:K
    indice=randperm(l);
    for i=1:l
        i=indice(i);
        if(y(i)*(x(i,:)*w1)<=0)
            w1=w1+y(i)*x(i,:)';
            k=k+1;
        end
    end
end

% HyperPlan associ? 
figure(1)
hold on
title('Systeme OR perceptron')
xlabel('x1')
ylabel('x2')
t=-5:0.01:5;
plot(t,-(w1(1)/w1(2))*t-w1(3)/w1(2));
for j=1:l
    if(y(j)==1)
        plot(x(j,1),x(j,2),'b+');
    else
        plot(x(j,1),x(j,2),'r*');
    end
end
hold off
grid on 


%%

l=4;
n=2;
% Systeme AND
x1=[0 0 1];
x4=[0 1 1];
x3=[1 0 1];
x2=[1 1 1];
x=[x1;x2;x3;x4];
y=[-1 -1 -1 1];

% Algo perceptron
w2=zeros(n+1,1);
k=0;
K=57;
for compteur=1:K
    indice=randperm(l);
    for i=1:l
        i=indice(i);
        if(y(i)*(x(i,:)*w2)<=0)
            w2=w2+y(i)*x(i,:)';
            k=k+1;
        end
    end
end

% HyperPlan associ? 
figure(2)
t=-5:0.01:5;
plot(t,-(w2(1)/w2(2))*t-w2(3)/w2(2));
hold on
title('Systeme AND perceptron')
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

%% Methode moindre carr?s



% l le nombre de donn?es, n la dimension des donn?es
l=4;
n=2;

% Systeme OR
x1=[0 0 1];
x2=[0 1 1];
x3=[1 0 1];
x4=[1 1 1];
x=[x1;x2;x3;x4];
y=[-1 1 1 1];

% Algo moindre carr?s
w_mc_or=zeros(n+1,1);
alpha=1;
K=67;
for compteur=1:K
    indice=randperm(l);
    for i=1:l
        i=indice(i);
        g=-1/l*(y(i)-w_mc_or'*x(i,:)')*x(i,:)';
        w_mc_or=w_mc_or -alpha*g;
    end
    alpha=alpha/compteur;
end

% HyperPlan associ? 
figure(3)
t=-5:0.01:5;
plot(t,-(w_mc_or(1)/w_mc_or(2))*t-w_mc_or(3)/w_mc_or(2));
hold on
title('Systeme OR avec moindre carres')
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

% Systeme AND
x1=[0 0 1];
x4=[0 1 1];
x3=[1 0 1];
x2=[1 1 1];
x=[x1;x2;x3;x4];
y=[-1 -1 -1 1];

% Algo moindre carr?s
w_mc_and=zeros(n+1,1);
alpha=1;
K=6;
for compteur=1:K
    indice=randperm(l);
    for i=1:l
        i=indice(i);
        g=-1/l*(y(i)-w_mc_and'*x(i,:)')*x(i,:)';
        w_mc_and=w_mc_and -alpha*g;
    end
    alpha=alpha/compteur;
end

% HyperPlan associ? 
figure(4)
t=-5:0.01:5;
plot(t,-(w_mc_and(1)/w_mc_and(2))*t-w_mc_and(3)/w_mc_and(2));
hold on
title('Systeme AND moindre carres')
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

%% Donn?es s?parables
clear all
% l le nombre de donn?es, n la dimension des donn?es
l=20;
n=2;

% Systeme avec 2 classes separables 

x=[0 1 1; 1 1 1; 2 3 1; 3 1 1;4 5 1;6 1 1;7 2  1;6 2 1;2 2 1; 3 3 1; 8 9 1;10 9 1; 12 8 1;10 10 1; 11 11 1;7 13 1;9 12 1;13 10 1;13 11 1;9 9 1];
y=[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 1 1 1 1 1 1 1 1];

% Algo perceptron
wp=zeros(n+1,1);
k=0;
K=670;
for compteur=1:K
    indice=randperm(l);
    for i=1:l
        i=indice(i);
        if(y(i)*(x(i,:)*wp)<=0)
            wp=wp+y(i)*x(i,:)';
            k=k+1;
        end
    end
end

% HyperPlan associ? 
figure(5)
hold on
title('2 classes separables avec perceptron')
xlabel('x1')
ylabel('x2')
t=-20:0.001:20;
f=-wp(1)/wp(2)*t-wp(3)/wp(2);
plot(t,f);
for j=1:l
    if(y(j)==1)
        plot(x(j,1),x(j,2),'b+');
    else
        plot(x(j,1),x(j,2),'r*');
    end
end
hold off
grid on 

% Algo moindre carr?s
w_mc=zeros(n+1,1);
alpha=1;
K=6;
for compteur=1:K
    indice=randperm(l);
    for i=1:l
        i=indice(i);
        g=-1/l*(y(i)-w_mc'*x(i,:)')*x(i,:)';
        w_mc=w_mc -alpha*g;
    end
    alpha=alpha/compteur;
end

% HyperPlan associ? 
figure(4)
t=-20:0.01:20;
plot(t,-(w_mc(1)/w_mc(2))*t-w_mc(3)/w_mc(2));
hold on
title('2 classes separables avec moindre carrees')
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

