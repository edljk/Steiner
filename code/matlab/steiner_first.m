close all;
clear;

%addpath('../common/');

% Target points
q=4;
tt = linspace(0,2*pi, q+1);
targetp = [cos(tt(1:end-1))',sin(tt(1:end-1))']*.95;
%targetp(end,:) = [0 0];

% Mesh
np = 20;
x = linspace(-1,1,np);
y = linspace(-1,1,np);
[X,Y] = meshgrid(x,y);

X = -1 + 2*rand(size(X));
Y = -1 + 2*rand(size(Y));

X = [X(:);targetp(:,1)]; 
Y = [Y(:);targetp(:,2)];
dt = delaunayTriangulation(X(:),Y(:));

p = dt.Points;
np = size(p,1);

ep = dt.edges;
em = [ep(:,2), ep(:,1)];
e = [ep; em];
ne = size(e,1);

l = sqrt(sum((p(e(:,1),:)- p(e(:,2),:)).^2, 2));
%l = l + rand(size(l))*0.01;

A = sparse(e(:,1), 1:ne, ones(ne,1), np, ne) + ...
    sparse(e(:,2), 1:ne, -ones(ne,1), np, ne);

L = [speye(ne,ne), -speye(ne,ne)];

b = zeros(np, q);
idx = 0;

targets = np-length(targetp)+1:np;
b(targets,:) = eye(length(targetp));
b(end,:) = -ones(1,q); % sink
b(end,end) = 0;

% primals
u = zeros(ne, q);
s = zeros(ne,1);
t = zeros(ne,1);

% duals
psi = zeros(ne,q);
phi = zeros(ne,q);
xi = zeros(np,q);

maxiter = 5000;
Lip = 5;
tau = 0.1/Lip;
sigma = 1/tau/Lip^2;

for iter=1:maxiter
  
  psi_ = psi;
  phi_ = phi;
  xi_ = xi;
  
  for j=1:q
    psi(:,j) = max(0, psi(:,j) + sigma*(L*[u(:,j);s]));
    phi(:,j) = min(0, phi(:,j) + sigma*(L*[u(:,j);t]));
    xi(:,j) = xi(:,j) + sigma*(A*u(:,j) - b(:,j));
  end
  
  psi_ = 2*psi - psi_;
  phi_ = 2*phi - phi_;
  xi_ = 2*xi - xi_;
  
  for j=1:q
    
    tmp_s = L'*psi_(:,j);
    tmp_t = L'*phi_(:,j);
    
    u(:,j) = u(:,j) - tau*(tmp_s(1:ne) + tmp_t(1:ne) + A'*xi_(:,j));
    s = s - tau*(l + tmp_s(ne+1:end));
    t = t - tau*(-l + tmp_t(ne+1:end));
  end
  
  if mod(iter, 1000) == 0
    feas = norm(A*u-b, 'fro');
    fprintf('iter = %04d, feas = %f\n', iter, feas);
    
    %sfigure(1);
    hold on;
    
    v = 1-max(0, s-t);
    minmaxmean(v)
    for i=1:ne      
      v = 1-max(0, s(i)-t(i));
      if v > 0.1
        plot(p(e(i,:),1), p(e(i,:),2), 'Color',ones(1,3)*v);
      end
    end
    
    for i=1:q
      plot(p(targets(i), 1), p(targets(i),2), 'o', 'Markersize', 5, 'Linewidth', 2);
    end
    hold off;
    axis equal
    drawnow;
  end
end






