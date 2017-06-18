clear all
close all 
clc

A = imread('images/low_risk_1.jpg');
[N1, N2, N3] = size(A);
N = N1*N2;  
B = double(reshape(A,N,N3));
clusters = 4; 
[N, M] = size(B);
Bnew = zeros(N, M);

rng('default');
% Initial guess centroids
x_k = 200 * rand(clusters,M);
distance = zeros(1,clusters);
count = 10; 
for nit=1:count
    dec=zeros(N,1);
    for i=1:N
        for k = 1:clusters
            distance(k) = (norm(B(i,:)-x_k(k,:)))^2;
        end
        % Finding the index at the minimum distance:
        [Y,I] = min(distance);
        % Pixel B(i,:) is given to cluster I
        dec(i)=I;
    end
    xnew=zeros(clusters,3);
    for kk=1:clusters
        indexes=find(dec==kk);
        xnew(kk,:)=floor(mean(B(indexes,:)));
        Bnew(indexes,:)=ones(length(indexes),1)*xnew(kk,:);
    end
    x_k=xnew;
end
% idx = kmeans(B,clusters); 
% x_k = 200 * rand(clusters, M);
% 
% for i=1:N
%     if idx(i) == 1
%         Bnew2(i,:) = x_k(1,:);
%     elseif idx(i) == 2 
%         Bnew2(i,:) = x_k(2,:);
%     elseif idx(i) == 3 
%         Bnew2(i,:) = x_k(3,:);
%     elseif idx(i) == 4 
%         Bnew2(i,:) = x_k(4,:);        
%     end        
% end
Bnew=floor(Bnew);
Anew=reshape(uint8(Bnew),N1,N2,N3);
figure(),subplot(1,2,1),imshow(A),title('Low risk 1'),
subplot(1,2,2),imshow(Anew),title('Clustered image'),
imwrite(Anew, 'results/low_risk_1_R.jpg');