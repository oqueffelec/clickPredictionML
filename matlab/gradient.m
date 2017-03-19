% Soit xj une instance d'une pub avec ses differents features et yj le
% label avec n=10

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
    n=0.001;
    w_old=ones(10,1);

for j=1:10
    h=(y(j)-x(:,j)'*w_old)*x(:,j);
    w_new=w_old-h*n; 
    w_old=w_new;
end

% Fonction f de prediction pour les 10 instances

f1=x(:,1)'*w_new;
f2=x(:,2)'*w_new;
f3=x(:,3)'*w_new;
f4=x(:,4)'*w_new;
f5=x(:,5)'*w_new;
f6=x(:,6)'*w_new;
f7=x(:,7)'*w_new;
f8=x(:,8)'*w_new;
f9=x(:,9)'*w_new;
f10=x(:,10)'*w_new;

% Si f est positive y_chap=1 sinon y_chap=-1... Dans mon cas les 10 f sont
% positives car j'ai pris des valeurs completement random pour les
% instances donc il n'y aucune logique au resultat... C'etait juste pour
% illustrer la d?marche




