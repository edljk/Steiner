close all;
clear;

rng(42);
addpath('../common/');

mex CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" shrink_osc_mex.cpp
mex CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" shrink_osc2_mex.cpp

maxiter = 100000;
check = 1000;

%% 3 points
if 0
  q=3;
  targetp = [0,1; -sin(2*pi/3),-1/2; sin(2*pi/3),-1/2];
end

%% 4 points
if 0
  targetp = [-1, -1; 1, -1; -1, 1; 1, 1];
  q=4;
end

%% 5 points
if 0
  q=5;
  tt = linspace(0,2*pi, q+1);
  targetp = [cos(tt(1:end-1))',sin(tt(1:end-1))'];
end

%% 6 points and one in the middle
if 0
  q=6;
  tt = linspace(0,2*pi, q+1);
  targetp = [cos(tt(1:end-1))',sin(tt(1:end-1))'];
  q = q+1;
  targetp = [targetp; 0,0];
end

%% 2 rows with 3 points
if 0
  targetp = [-1,-0.5; 0,-0.5; 1, -0.5;
             -1, 0.5; 0, 0.5; 1,  0.5];
  q=6;
end


%% 2 rows with 5 points
if 1
  targetp = [-1,-0.25; -0.5,-0.25; 0,-0.25; 0.5,-0.25; 1,-0.25;
             -1, 0.25; -0.5, 0.25; 0, 0.25; 0.5, 0.25; 1, 0.25];
  q=10;
end

%% Compute mesh: regular grid + noise
n = 20;
[X,Y] = meshgrid(linspace(-1,1,n), linspace(-1,1,n));

% add noise to enforce uniqueness
X = X + randn(n,n)/(4*n);
Y = Y + randn(n,n)/(4*n);

%% Add target points
p = [[X(:);targetp(:,1)], [Y(:);targetp(:,2)]];
np = size(p,1);

%% compute edges
nn = 100;
idx = knnsearch(p,p,'K',nn+1);
e = reshape(cat(3,repmat(1:length(p),nn,1)',idx(:,2:end)),[],2);
e(e(:,1)>e(:,2),:) = []; % only one edge!

ne = size(e,1);
l = sqrt(sum((p(e(:,1),:)- p(e(:,2),:)).^2, 2));

%% Flow constraint
A = sparse(e(:,1), 1:ne, ones(ne,1), np, ne) + ...
    sparse(e(:,2), 1:ne, -ones(ne,1), np, ne);
 
%% Source / sink points
b = zeros(np, q);
targets = np-length(targetp)+1:np;
b(targets,:) = eye(length(targetp));
b(end,:) = -ones(1,q); 
b(end,end) = 0;

%% Varables
u = zeros(ne, q);
xi = zeros(np,q);

%% Preconditioning
fact = 1;
tau = fact./sum(abs(A), 1)';
sigma = 1./sum(abs(A), 2)/fact;
rho = 1.9;
dykstra_iter = 1;
for iter=1:maxiter
  
  xi_old = xi;
  u_old  = u;
  
  % dual update
  for j=1:q
    xi(:,j) = xi(:,j) + sigma.*(A*u(:,j) - b(:,j));
  end
    
  xi_ = 2*xi - xi_old;
  
  % primal update
  for j=1:q
    u(:,j) = u(:,j) - tau.*(A'*xi_(:,j));
  end
  
  %[u, dykstra_iter] = shrink_osc_mex(u, tau.*l);  
  [u] = shrink_osc2_mex(u, tau.*l);
  
  % overrelaxation
  xi = (1-rho)*xi_old + rho*xi;
  u = (1-rho)*u_old + rho*u;

  if mod(iter, check) == 0
    
    div = max(max(abs(A*u-b)));
       
    v = max(u,[],2) - min(u,[],2);
    v_max = max(v(:));
    v_min = min(v(:));
    
    f = v'*l;
    
    fprintf('iter = %04d, length = %f, ||div||_infty = %f, v_min = %f, v_max = %f\n\n', ...
    iter, f, div, v_min, v_max);
    
    sfigure(1);
    plot(X,Y, '.b', 'Markersize', 1); hold on;
    for i=1:q
      plot(p(targets(i), 1), p(targets(i),2), 'ro', 'Markersize', 5, ...
      'Linewidth', 2);
    end
    tree = find(v > 0.25);
    for i=1:length(tree)
      c = max(0, min(1, 1-v(tree(i))));
      plot(p(e(tree(i),:),1), p(e(tree(i),:),2), 'Color',...
        ones(1,3)*c, 'Linewidth', 1);
    end
    hold off;
    
    xlim([-1, 1]);
    ylim([-1, 1]);
    axis equal;
    drawnow;
        
    filename = sprintf('out/%08d.png', iter/check);
    set(gcf, 'PaperPosition', [0 0 7 7]);
    set(gcf, 'PaperSize', [7 7]);
    saveas(gcf, filename, 'png');
    
  end
end