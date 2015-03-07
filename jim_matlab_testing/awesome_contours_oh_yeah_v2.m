function awesome_contours_oh_yeah_v2(frame_values)
%
%   too much Despicable Me watching ...
%

%Problem frames:
%1200 - ends are a bit messed up


file_path = 'C:\Users\RNEL\Google Drive\open_worm\example_data\example_contour_and_skeleton_info.mat';

%Intersection is a dangerous game because of the problem of being very
%close ...



h = load(file_path);

%frame_value = 1;
for iFrame = frame_values %1:100:4642
s1 = h.all_vulva_contours{iFrame};
s2 = h.all_non_vulva_contours{iFrame};

if size(s1,1) > size(s2,1)
    [s1,s2] = deal(s2,s1);
end

if isempty(s1)
    continue
end

vc_raw = s1;
nvc_raw = s2;

tic

s1(:,1) = sgolayfilt(s1(:,1),3,sl.math.roundToOdd(size(s1,1)/12));
s1(:,2) = sgolayfilt(s1(:,2),3,sl.math.roundToOdd(size(s1,1)/12));

s2(:,1) = sgolayfilt(s2(:,1),3,sl.math.roundToOdd(size(s2,1)/12));
s2(:,2) = sgolayfilt(s2(:,2),3,sl.math.roundToOdd(size(s2,1)/12));


%{
%A plot of the smoothed values ...
hold on
plot(vc(2:end-1,1),vc(2:end-1,2),'ko')

plot(nvc(2:end-1,1),nvc(2:end-1,2),'ko')
hold off
%}





[norm_x,norm_y] = h__computeNormalVectors(s1);





%TODO: Allow downsampling if the # of points is rediculous
%This operation gives us a matrix that is len(s1) x len(s2)
dx_across = bsxfun(@minus,s1(:,1),s2(:,1)');
dy_across = bsxfun(@minus,s1(:,2),s2(:,2)');
d_across  = sqrt(dx_across.^2 + dy_across.^2);
dx_across = dx_across./d_across;
dy_across = dy_across./d_across;


[dp_values,match_I] = h__getMatches(s1,s2,norm_x,norm_y,dx_across,dy_across);


toc

clf
subplot(2,3,[1 2 4 5])
hold on

%Raw
plot(vc_raw(:,1),vc_raw(:,2),'r.')
plot(nvc_raw(:,1),nvc_raw(:,2),'b.')

%Smooth
plot(s1(2:end-1,1),s1(2:end-1,2),'ro')
plot(s2(2:end-1,1),s2(2:end-1,2),'bo')

for iPlot = 1:size(s1,1)
    I2 = match_I(iPlot);
    if iPlot == 5 %Start a bit in so we see it
        c = 'm';
    elseif dp_values(iPlot) > 0.99
        c = 'g';
    else
        c = 'k';
    end
    plot([s1(iPlot,1) s2(I2,1)],[s1(iPlot,2) s2(I2,2)],c)
end

hold off
axis equal

subplot(2,3,3)
plot(dp_values,'o-')
set(gca,'ylim',[0 1])

subplot(2,3,6)
I_row = (1:size(s1,1))';
I_col = match_I;
I_linear = sub2ind(size(d_across),I_row,I_col);
plot(d_across(I_linear),'o-')

title(sprintf('iFrame %d',iFrame))

pause

end





end

function [dp_values,match_I] = h__getMatches(s1,s2,norm_x,norm_y,dx_across,dy_across)


n_s1 = size(s1,1);
match_I = zeros(n_s1,1);
match_I(1) = 1;
match_I(end) = size(s2,1);
dp_values = ones(size(s1,1),1);


%We might want to dynamically adjust this ...
cur_right_offset = 50;
rb = 10;
max_right = size(s2,1);

for I = 2:n_s1-1;
    
    lb = I - 30; %left boundary
    if lb < 1
        lb = 1;
    end
    %rb = I + cur_right_offset; %right boundary
    if rb > max_right
        rb = max_right;
    end
    
    
    [abs_dp_value,dp_I] = h__getProjectionIndex(norm_x(I),norm_y(I),dx_across(I,lb:rb),dy_across(I,lb:rb),lb);
    
    if dp_I == rb && rb < max_right;
       rb = rb + 10;
       if rb > max_right
          rb = max_right; 
       end
       [abs_dp_value,dp_I] = h__getProjectionIndex(norm_x(I),norm_y(I),dx_across(I,lb:rb),dy_across(I,lb:rb),lb); 
    end
    
    rb = dp_I + 10;
    
    
    dp_values(I) = abs_dp_value;
    match_I(I) = dp_I;
    
end



end

function [norm_x,norm_y] = h__computeNormalVectors(data)

dx = gradient(data(:,1));
dy = gradient(data(:,2));
dx_norm = dy;
dy_norm = -dx;

vc_d_magnitude = sqrt(dx_norm.^2 + dy_norm.^2);

norm_x = dx_norm./vc_d_magnitude;
norm_y = dy_norm./vc_d_magnitude;


end

function [abs_dp_value,I] = h__getProjectionIndex(vc_dx_ortho,vc_dy_ortho,dx_across_worm,dy_across_worm,left_I)


% nvc_local = nvc(nvc_indices_use,:);
%
% dx_across_worm = cur_point(1) - nvc_local(:,1);
% dy_across_worm = cur_point(2) - nvc_local(:,2);
%
% d_magnitude = sqrt(dx_across_worm.^2+dy_across_worm.^2);
%
% dx_across_worm = dx_across_worm./d_magnitude;
% dy_across_worm = dy_across_worm./d_magnitude;


%SPEED: Compute normalized distances for all pairs ...
%Might need to downsample

dp = dx_across_worm*vc_dx_ortho + dy_across_worm*vc_dy_ortho;

[abs_dp_value,max_dp_I] = max(abs(dp));

I = left_I + max_dp_I - 1;

end

