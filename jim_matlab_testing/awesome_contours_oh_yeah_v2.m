function awesome_contours_oh_yeah_v2(frame_values)
%
%   too much Despicable Me watching ...
%
%   Algorithm:
%   ----------
%   Find which point when taking the line from the current point to that
%   opposite point and the normal of the current point, makes the 2
%   parallel.
%
%   x   c   x
%      | |      left | is the normal for c, based on its gradient
%        |      right | is the line between c and the middle y
%        |      find which y makes the 2 lines most parallel (dot product maximized
%   y   y   y
%
%
%
%   Status:
%   -------
%   1) algorithm doesn't work on non parallel surfaces, consider a diamond
%    /|\
%   / | \
%   \ | /
%    \|/
%       This practically has an effect at the ends and needs to be fixed.
%      Interpolation of good values might fix this problem.
%
%   2) FIXED Points are not guaranteed to be ordered, so a dot product would
%   need to be computed for subsequent points to look for reversals.
%
%   i.e., this would be fine for midpoints - no backtracking
%   1 2
%       3
%         4
%
%   this would not: point 3 would need to be removed ...
%   1   3 2 4
%
%   3) Spurs are not handled. It would be good to get an example of this
%   since I think we could handle this.
%
%   Interesting Frames:
%   -------------------
%   11 - good example of ends not being nicely shaped
%   261 - failure when the back/forth search is not pretty wide (0.3
%   instead of 0.2)
%   3601 - coiling that fails with dot product unless a search is done
%       for multiple peaks and the smaller width chosen
%

%Original example contour & skeleton data from:
%

%Problem frames:
%1200 - ends are a bit messed up


file_path = 'C:\Users\RNEL\Google Drive\open_worm\example_data\example_contour_and_skeleton_info.mat';
fp2 = 'C:\Users\RNEL\Google Drive\open_worm\example_data\example_video_norm_worm.mat';

%Intersection is a dangerous game because of the problem of being very
%close ...

h2 = load(fp2);

nw_widths = h2.s.widths;
nw_sx = h2.s.x;
nw_sy = h2.s.y;

h = load(file_path);

%frame_value = 1;
for iFrame = frame_values %1:100:4642
    s1 = h.all_vulva_contours{iFrame};
    s2 = h.all_non_vulva_contours{iFrame};
    
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
    
    
    %TODO: Allow downsampling if the # of points is rediculous
    %This operation gives us a matrix that is len(s1) x len(s2)
    dx_across = bsxfun(@minus,s1(:,1),s2(:,1)');
    dy_across = bsxfun(@minus,s1(:,2),s2(:,2)');
    d_across  = sqrt(dx_across.^2 + dy_across.^2);
    dx_across = dx_across./d_across;
    dy_across = dy_across./d_across;
    
    
    %With 0.2 messing up on 261
    P_BACK = 0.3;
    P_FORWARD = 0.3;
    %all s1 matching to s2
    %-------------------------
    [left_I,right_I] = h__getBounds(size(s1,1),size(s2,1),P_BACK,P_FORWARD);
    [norm_x,norm_y]  = h__computeNormalVectors(s1);
    
    [dp_values1,match_I1] = h__getMatches(s1,s2,norm_x,norm_y,dx_across,dy_across,d_across,left_I,right_I);
    
    END_S1_WALK = 40; %TODO: Make a percentage ...
    [p1_I,p2_I] = h__getPartnersViaWalk(1,END_S1_WALK,1,100,d_across,s1,s2);
    
    match_I1(p1_I) = p2_I;
    
    keep_mask = false(1,length(match_I1));
    keep_mask(p1_I) = true;
    
    n_s1 = length(s1);
    n_s2 = length(s2);
    END_S1_WALK_BACKWARDS = n_s1 - 40;
    END_S2_WALK_BACKWARDS = n_s2 - 60;
    
    
    [p1_I,p2_I] = h__getPartnersViaWalk(n_s1,END_S1_WALK_BACKWARDS,n_s2,END_S2_WALK_BACKWARDS,d_across,s1,s2);
    match_I1(p1_I) = p2_I;
    keep_mask(p1_I) = true;
    
    %anything in between we'll use the projection appproach
    keep_mask(END_S1_WALK+1:END_S1_WALK_BACKWARDS-1) = true;
    
    %Always keep ends
    keep_mask(1)   = true; 
    keep_mask(end) = true;
    
    match_I1(p1_I) = p2_I;
    match_I1(1) = 1;

    
    s1_x = s1(keep_mask,1);
    s1_y = s1(keep_mask,2);
    s1_px = s2(match_I1(keep_mask),1);
    s1_py = s2(match_I1(keep_mask),2);
    
% %     mid_x = 0.5*(s1_x + s1_px);
% %     mid_y = 0.5*(s1_y + s1_py);
    % dx = diff(mid_x);
    % dy = diff(mid_y);
    %
    % dm = sqrt(dx.^2+dy.^2
    
    
    
    %keyboard
    
    
    %Causes problems with frame 41
    %Use a smoothing algorithm instead ...
    % % % %remove_mask = h__getBackwardsRemoveMask(mid_x,mid_y);
    % % %
    % % % %TODO: Make % of worm
    % % % % remove_mask(2:30) = true;
    % % % % remove_mask(end-30:end-1) = true;
    % % %
    % % %
    % % % dp_values1(remove_mask) = [];
    % % % s1_x(remove_mask) = [];
    % % % s1_y(remove_mask) = [];
    % % % s1_px(remove_mask) = [];
    % % % s1_py(remove_mask) = [];
    
    
    mid_x = 0.5*(s1_x + s1_px);
    mid_y = 0.5*(s1_y + s1_py);
    widths1 = sqrt((s1_px-s1_x).^2 + (s1_py - s1_y).^2); %widths
    cum_dist1 = h__getSkeletonDistance(mid_x,mid_y);
    
    
    
    
    % new_d = linspace(cum_dist1(1),cum_dist1(end),200);
    %
    % f = csape(cum_dist1,mid_x);
    % fx = fnval(f,new_d);
    % f = csape(cum_dist1,mid_y);
    % fy = fnval(f,new_d);
    
    % % subplot(2,1,1)
    % % plot(cum_dist1,mid_x,'o')
    % %
    % % hold all
    % % plot(new_d,fx);
    % % hold off
    % %
    % % subplot(2,1,2)
    % % plot(cum_dist1,mid_y,'o')
    % % hold all
    % % plot(new_d,fy);
    % % hold off
    
    %all s2 matching to s1
    %---------------------
% % % %     [left_I,right_I] = h__getBounds(size(s2,1),size(s1,1),P_BACK,P_FORWARD);
% % % %     [norm_x,norm_y]  = h__computeNormalVectors(s2);
% % % %     
% % % %     [dp_values2,match_I2] = h__getMatches(s2,s1,norm_x,norm_y,dx_across',dy_across',d_across',left_I,right_I);
    
    %toc
    %Plotting Results
    %-------------------
    if true
        toc
        %     plot_s2_match = false;
        %
        %     if plot_s2_match
        %         dp_values = dp_values2;
        %         match_I   = match_I2;
        %         [s1,s2]   = deal(s2,s1);
        %         d_across  = d_across';
        %     else
        dp_values = dp_values1;
        %     end
        
        
        
        clf
        subplot(2,3,[1 2 4 5])
        hold on
        
        %Raw
        plot(vc_raw(:,1),vc_raw(:,2),'r.')
        plot(nvc_raw(:,1),nvc_raw(:,2),'b.')
        
        %Smooth
        plot(s1(2:end-1,1),s1(2:end-1,2),'ro')
        plot(s2(2:end-1,1),s2(2:end-1,2),'bo')
        
        plot(mid_x,mid_y,'d-')
        %     plot(fx,fy,'-k')
        
        
        for iPlot = 1:length(s1_x)
            %I2 = match_I(iPlot);
            
            if iPlot == 5 %Start a bit in so we see it
                c = 'm';
            elseif abs(dp_values(iPlot)) > 0.99
                c = 'g';
            else
                c = 'k';
            end
            x = [s1_x(iPlot) s1_px(iPlot)];
            y = [s1_y(iPlot) s1_py(iPlot)];
            %midpoint = [0.5*(x(1)+x(2)),0.5*(y(1)+y(2))];
            plot(x,y,c)
            %plot(midpoint(1),midpoint(2),'k.')
        end
        
        %     for iPlot = 1:size(s2,1)
        %         I1 = match_I2(iPlot);
        %         x = [s1(I1,1) s2(iPlot,1)];
        %         y = [s1(I1,2) s2(iPlot,2)];
        %         midpoint = [0.5*(x(1)+x(2)),0.5*(y(1)+y(2))];
        %         plot(midpoint(1),midpoint(2),'ko')
        %     end
        
        plot(nw_sx(:,iFrame),nw_sy(:,iFrame),'x','Color',[0.3 0.3 0.3])
        
        hold off
        axis equal
        
        subplot(2,3,3)
        
        plot(dp_values,'o-')
        set(gca,'ylim',[-1 -0.5])
        
        
        %Width should really be plotted as a function of distance along the skeleton
        
        
        cum_dist = h__getSkeletonDistance(mid_x,mid_y);
        
        subplot(2,3,6)
        plot(cum_dist./cum_dist(end),widths1,'r.-')
        hold on
        plot(linspace(0,1,49),nw_widths(:,iFrame),'g.-')
        hold off
        
        title(sprintf('iFrame %d',iFrame))
        
        if length(frame_values) > 1
            pause
        end
        
    end
end





end

function remove_mask = h__getBackwardsRemoveMask(mid_x,mid_y)
%
%   We want the dot product of vectors to keep moving forward not back
%
%   We're using a cutoff of 0 here, which means anything smaller than a right
%   turn is allowed.
%

remove_mask = false(1,length(mid_x));

last_dx = mid_x(2)-mid_x(1);
last_dy = mid_y(2)-mid_y(1);
dm = sqrt(last_dx.^2 + last_dy.^2);
last_dx = last_dx./dm;
last_dy = last_dy./dm;

last_valid_point = 2; %We keep points 1 & 2 to initialize
for iPoint = 3:length(mid_x)
    cur_dx = mid_x(iPoint)-mid_x(last_valid_point);
    cur_dy = mid_y(iPoint)-mid_y(last_valid_point);
    dm = sqrt(cur_dx.^2 + cur_dy.^2);
    cur_dx = cur_dx./dm;
    cur_dy = cur_dy./dm;
    
    dp = cur_dx*last_dx + cur_dy*last_dy;
    if dp > 0
        last_dx = cur_dx;
        last_dy = cur_dy;
        last_valid_point = iPoint;
    else
        remove_mask(iPoint) = true;
    end
end
end

function cum_dist = h__getSkeletonDistance(mid_x,mid_y)
dx = diff(mid_x);
dy = diff(mid_y);
d = [0; sqrt(dx.^2+dy.^2)];
cum_dist = cumsum(d);
end

function [left_I,right_I] = h__getBounds(n1,n2,p_left,p_right)

pct = linspace(0,1,n1);
left_pct = pct - p_left;
right_pct = pct + p_right;

left_I = floor(left_pct*n2);
right_I = ceil(right_pct*n2);
left_I(left_I < 1) = 1;
right_I(right_I > n2) = n2;
end

function [dp_values,match_I] = h__getMatches(s1,s2,norm_x,norm_y,dx_across,dy_across,d_across,left_I,right_I)


n_s1 = size(s1,1);
match_I = zeros(n_s1,1);
match_I(1) = 1;
match_I(end) = size(s2,1);
dp_values = ones(size(s1,1),1);
dp_values(1) = -1;
dp_values(end) = -1;

for I = 2:n_s1-1;
    
    lb = left_I(I);
    rb = right_I(I);
    
    [abs_dp_value,dp_I] = h__getProjectionIndex(norm_x(I),norm_y(I),dx_across(I,lb:rb),dy_across(I,lb:rb),lb,d_across(I,lb:rb));
    
    dp_values(I) = abs_dp_value;
    match_I(I)   = dp_I;
end

end

function [norm_x,norm_y] = h__computeNormalVectors(data)

dx = gradient(data(:,1));
dy = gradient(data(:,2));

%This approach gives us -1 for the projection
%We could also use:
%dx_norm = -dy;
%dy_norm = dx;
%
%and we would get 1 for the projection
dx_norm = dy;
dy_norm = -dx;

vc_d_magnitude = sqrt(dx_norm.^2 + dy_norm.^2);

norm_x = dx_norm./vc_d_magnitude;
norm_y = dy_norm./vc_d_magnitude;


end

function [dp_value,I] = h__getProjectionIndex(vc_dx_ortho,vc_dy_ortho,dx_across_worm,dy_across_worm,left_I,d_across)


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

%               < right                         < left
%possible = [dp(1:end-1) < dp(2:end) false] & [false dp(2:end) < dp(1:end-1)];
possible = (dp(2:end-1) < dp(3:end)) & (dp(2:end-1) < dp(1:end-2));

%What I really need are local peaks and then getting the best local
%peak with the smallest width




Ip = find(possible);
if length(Ip) == 1
    dp_I = Ip+1;
    dp_value = dp(dp_I);
elseif length(Ip) > 1
    [~,temp_I] = min(d_across(Ip));
    dp_I     = Ip(temp_I)+1;
    dp_value = dp(dp_I);
else
    [dp_value,dp_I] = min(dp);
end

I = left_I + dp_I - 1;

end

function [p1_I,p2_I] = h__getPartnersViaWalk(s1,e1,s2,e2,d,xy1,xy2)
%
%   s1: start index for side 1
%   e1: end index for side 1
%
%   d :
%       distance from I1 to I2 is d(I1,I2)
%
%   d1 : [n x 2]
%       x,y pairs for side 1
%
%
%


%TODO: remove hardcode
p1_I = zeros(1,200);
p2_I = zeros(1,200);

c1 = s1; %current 1 index
c2 = s2; %current 2 index
cur_p_I = 0; %current pair index


while c1 ~= e1 && c2 ~= e2
    cur_p_I = cur_p_I + 1;
    
    if e1 < s1
        next1 = c1-1;
        next2 = c2-1;        
    else
        next1 = c1+1;
        next2 = c2+1;
    end
    
    v_n1c1 = xy1(next1,:) - xy1(c1,:);
    v_n2c2 = xy2(next2,:) - xy2(c2,:);
    
    d_n1n2 = d(next1,next2);
    d_n1c2 = d(next1,c2);
    d_n2c1 = d(c1,next2);
    
    
    if d_n1c2 == d_n2c1 || (d_n1n2 <= d_n1c2 && d_n1n2 <= d_n2c1)
        %Advance along both contours
        
        p1_I(cur_p_I) = next1;
        p2_I(cur_p_I) = next2;
        
        c1 = next1;
        c2 = next2;
        
    elseif all((v_n1c1.*v_n2c2) > -1)
        %contours go similar directions
        %follow smallest width
        if d_n1c2 < d_n2c1
            %consume smaller distance, then move the base of the vector
            %further forward
            p1_I(cur_p_I) = next1;
            p2_I(cur_p_I) = c2;
            
            %This bit always confuses me
            %c1  n1
            %
            %
            %c2  x  x  x  n2
            %
            %Advance c1 so that d_n2_to_c1 is smaller next time
            c1 = next1;
        else
            p1_I(cur_p_I) = c1;
            p2_I(cur_p_I) = next2;
            c2 = next2;
        end
    else
        
        if cur_p_I == 1
            prev_width = 0;
        else
            prev_width = d(p1_I(cur_p_I-1),p2_I(cur_p_I-1));
        end
        
        if (d_n1c2 > prev_width && d_n2c1 > prev_width)
            p1_I(cur_p_I) = next1;
            p2_I(cur_p_I) = next2;
            
            c1 = next1;
            c2 = next2;
        elseif d_n1c2 < d_n2c1
            p1_I(cur_p_I) = next1;
            p2_I(cur_p_I) = c2;
            c1 = next1;
        else
            p1_I(cur_p_I) = c1;
            p2_I(cur_p_I) = next2;
            c2 = next2;
        end
        
    end
    
end

p1_I(cur_p_I+1:end) = [];
p2_I(cur_p_I+1:end) = [];


end

