clc;clear;

gamma = 1;
xmax = 10;
vmax = 10;
amax = 10;
Nx = 2*xmax +1;
Nv = 2*vmax +1;
Na = 4*amax +1;
v = linspace(-vmax,vmax, Nv);      % discretize states into Nv and Nx segments
x = linspace(-xmax, xmax, Nx);
hv = 1/(2*vmax +1);             %v and x resolution
hx = 1/(2*xmax +1);
Ns = (1/hx)*(1/hv);                     % number of states


% s0 = [0,0];          %initial state s = [pos, vel]
A = linspace(-amax,amax,Na);       %Action space, acceleration, discretized 
P = zeros(Nx,Nv,Na,Nx,Nv);
R = zeros(Nx,Nv,Na,Nx,Nv);
R(:,:,:, xmax+1, vmax+1) = ones(Nx,Nv,Na);

for i=1:length(x)
    for j=1:length(v)
       zeta = [x(i),v(j)];
       for k=1:length(A)
           a = A(k);
           sp = f(zeta, a);                                   % deterministic dynamics, no distributions
           sp = bind(sp,xmax,vmax);
           
%            zetap = [round(sp(1)) round(sp(2))];               % 0th order Nearest Neighbor, k=1
%            P(i, j, k, zetap(1)+xmax+1, zetap(2)+vmax+1) = P(i, j, k, zetap(1)+xmax+1, zetap(2)+vmax+1) + 1; 
           
                                                              % 4-NN, k=4 nearest neighbor
           top_top = [ceil(sp(1)) ceil(sp(2))];               %4 nearest points
           top_bot = [ceil(sp(1)) floor(sp(2))];
           bot_top = [floor(sp(1)) ceil(sp(2))];
           bot_bot = [floor(sp(1)) floor(sp(2))];
           d1 = norm(sp-top_top);                              % distance to these 4 points
           d2 = norm(sp-top_bot);
           d3 = norm(sp-bot_top);
           d4 = norm(sp-bot_bot);
           if d1==0 && d2==0 && d3==0 && d4==0                     % if all zero then the point sp is exactly on a zeta
               P(i, j, k, top_top(1)+xmax+1, top_top(2)+vmax+1) = P(i, j, k, top_top(1)+xmax+1, top_top(2)+vmax+1) + 1;
           else
               pbar = [d1 d2 d3 d4]/sum([d1 d2 d3 d4]);                     % make sum equal 1
               P(i, j, k, top_top(1)+xmax+1, top_top(2)+vmax+1) = P(i, j, k, top_top(1)+xmax+1, top_top(2)+vmax+1) + pbar(1);
               P(i, j, k, top_bot(1)+xmax+1, top_bot(2)+vmax+1) = P(i, j, k, top_bot(1)+xmax+1, top_bot(2)+vmax+1) + pbar(2);
               P(i, j, k, bot_top(1)+xmax+1, bot_top(2)+vmax+1) = P(i, j, k, bot_top(1)+xmax+1, bot_top(2)+vmax+1) + pbar(3);
               P(i, j, k, bot_bot(1)+xmax+1, bot_bot(2)+vmax+1) = P(i, j, k, bot_bot(1)+xmax+1, bot_bot(2)+vmax+1) + pbar(4);
           end
           
       end
    end
end

%Policy iteration
policy = randi(length(A), Nx, Nv);
% policy = ones(Nx,Nv);
policy_old = policy.*2;                     %so the difference in policy is not less than epsilon on the first step
epsilon = 0.001;
value = zeros(Nx, Nv);
Q = zeros(Nx, Nv, Na);

valueVec = zeros(4,1);
qVec = zeros(4,1);
for n =1:100                         % number of iterations
    if abs(sum(sum(sum(policy-policy_old)))) < epsilon
        break;
    end
    %policy eval
    for i=1:length(x)               %loop through all states
        for j=1:length(v)
            zeta = [x(i), v(j)];
            sp = f(zeta, A(policy(i,j)));
            sp = bind(sp,xmax,vmax);
            
            spi = int64([sp(1)+xmax+1, sp(2)+vmax+1]);              % 0th order Nearest neighbor, 1-NN
            value(i,j) = squeeze(P(i,j,policy(i,j), spi(1), spi(2)))*(squeeze(R(i, j, policy(i,j), spi(1), spi(2))) + gamma*value(spi(1), spi(2)))';

%             top_top = [ceil(sp(1)) ceil(sp(2))];               %4 nearest points, attempt at 4-NN
%             top_bot = [ceil(sp(1)) floor(sp(2))];
%             bot_top = [floor(sp(1)) ceil(sp(2))];
%             bot_bot = [floor(sp(1)) floor(sp(2))];
%             tti = int64([top_top(1)+xmax+1, top_top(2)+vmax+1]);
%             tbi = int64([top_bot(1)+xmax+1, top_bot(2)+vmax+1]);
%             bti = int64([bot_top(1)+xmax+1, bot_top(2)+vmax+1]);
%             bbi = int64([bot_bot(1)+xmax+1, bot_bot(2)+vmax+1]);
%             valueVec(:,1) = [squeeze(P(i,j,policy(i,j), tti(1), tti(2)))*(squeeze(R(i, j, policy(i,j), tti(1), tti(2))) + gamma*value(tti(1), tti(2)))';
%                             squeeze(P(i,j,policy(i,j), tbi(1), tbi(2)))*(squeeze(R(i, j, policy(i,j), tbi(1), tbi(2))) + gamma*value(tbi(1), tbi(2)))';
%                             squeeze(P(i,j,policy(i,j), bti(1), bti(2)))*(squeeze(R(i, j, policy(i,j), bti(1), bti(2))) + gamma*value(bti(1), bti(2)))';
%                             squeeze(P(i,j,policy(i,j), bbi(1), bbi(2)))*(squeeze(R(i, j, policy(i,j), bbi(1), bbi(2))) + gamma*value(bbi(1), bbi(2)))'];
%             value(i,j) = sum(valueVec);
        end
    end
    %policy refinement
    for i=1:length(x)               %loop through all states
        for j=1:length(v)
            for k=1:length(A)
                zeta = [x(i), v(j)];
                sp = f(zeta, A(k));
                sp = bind(sp,xmax,vmax);
                
                spi = int64([sp(1)+xmax+1, sp(2)+vmax+1]);
                Q(i,j,k) = squeeze(P(i,j,k, spi(1), spi(2)))*(squeeze(R(i, j, k, spi(1), spi(2))) + gamma*value(spi(1), spi(2)))';
                
%                 top_top = [ceil(sp(1)) ceil(sp(2))];               %4 nearest points
%                 top_bot = [ceil(sp(1)) floor(sp(2))];
%                 bot_top = [floor(sp(1)) ceil(sp(2))];
%                 bot_bot = [floor(sp(1)) floor(sp(2))];
%                 tti = int64([top_top(1)+xmax+1, top_top(2)+vmax+1]);
%                 tbi = int64([top_bot(1)+xmax+1, top_bot(2)+vmax+1]);
%                 bti = int64([bot_top(1)+xmax+1, bot_top(2)+vmax+1]);
%                 bbi = int64([bot_bot(1)+xmax+1, bot_bot(2)+vmax+1]);
%                 qVec(:,1) = [squeeze(P(i,j,k, tti(1), tti(2)))*(squeeze(R(i, j, k, tti(1), tti(2))) + gamma*value(tti(1), tti(2)))';
%                             squeeze(P(i,j,k, tbi(1), tbi(2)))*(squeeze(R(i, j, k, tbi(1), tbi(2))) + gamma*value(tbi(1), tbi(2)))';
%                             squeeze(P(i,j,k, bti(1), bti(2)))*(squeeze(R(i, j, k, bti(1), bti(2))) + gamma*value(bti(1), bti(2)))';
%                             squeeze(P(i,j,k, bbi(1), bbi(2)))*(squeeze(R(i, j, k, bbi(1), bbi(2))) + gamma*value(bbi(1), bbi(2)))'];
%                 Q(i,j,k) = sum(valueVec);
            end
        end
    end
    policy_old = policy;
    [value_star, policy] = max(Q,[],3);
    
end

policy;
subplot(1,2,1)
surf(x,v,A(policy)')
xlabel('pos')
ylabel('vel')
subplot(1,2,2)
surf(x,v,value')
xlabel('pos')
ylabel('vel')



function [sp] = f(s, a)
    sp = [s(1)+s(2), s(2)+a];
end

function [sp] = bind(s,xmax,vmax)
    sp = s;
    if s(1) < -xmax
       sp(1) = -xmax;
    end
    if s(1) > xmax
       sp(1) = xmax;
    end
    if s(2) < -vmax
       sp(2) = -vmax;
    end
    if s(2) > vmax
       sp(2) = vmax;
    end
end