function [ myNotes ] = findNotes( K )
% This function gets a 2-D greyscale note sheet image as input and returns
% positions of each note in terms of staff line positions, figures of
% results after each stage
    K = imbinarize(im2double(rgb2gray(K)));                                % Binarization
    ANGLE=horizon(K, 0.1, 'hough');                                        % Hough transform
    K = 1-K;                                                               % Inverse
    J = imrotate(K,-1*ANGLE);                                              % Rotation

    [rows,cols] = size(J);

    M = sum(J,2);                                                          % Horizontal Projection
    mymax = max(M);                                                        % Array of horizontal projection maxima

    N = sum(J);                                                            % Clipping left and right
    zerocrossings = find(N==0);
    for i=1:round(cols/2)
        for j=1:length(zerocrossings)
            if zerocrossings(j)==i
                cols_zero1 = i;
                cols_zero2 = zerocrossings(j+1);
            end
        end
    end


    [PKS,LOCS] = findpeaks(M,'MinPeakHeight',mymax/2);                     % Maxima locations (staff lines)

    cnt = 0;                                                               % Finding thickness 
    for i = 1:length(M)
        if M(i) > mymax/2
            cnt = cnt + 1;
        end
    end
    thickness = round(cnt/length(PKS));


    for i=1:length(LOCS)                                                   % Line removal
        for j=3:cols-2
            temp = 0;
            for k = -thickness:1:thickness
            temp = temp + J(LOCS(i)+k,j);
            end
            if temp <= thickness
                for m = -thickness:1:thickness
                    J(LOCS(i)+m,j) = 0;
                end
            end
        end
    end

    J_yedek = J;                                                           % Matrix needed for finding connected components


    avgStaffDist = 0;                                                      % Average staff distance
    for i = 1:4
        avgStaffDist = avgStaffDist + LOCS(i+1)-LOCS(i);
    end
    avgStaffDist = round(avgStaffDist/4);

    clefs = imread('MusicSheets/Templates/gclef.png');
    clefs = 1-clefs;
    [rows_clefs,~] = size(clefs);
    clefs = imresize(clefs,avgStaffDist*8.5/rows_clefs);
    corr_mat_clefs = xcorr2(J_yedek,clefs);

    eighth = imread('MusicSheets/Templates/eight_d_24px.png');
    eighth = 1 - eighth;
    [rows8,~] = size(eighth);
    eighth = imresize(eighth,avgStaffDist*4/rows8);
    corr_mat = xcorr2(J_yedek,eighth);
    corr_thresh = xcorr2(eighth,eighth);
    [thresh_value,thresh_index] = max(corr_thresh(:));
    [mat_value,mat_index] = max(corr_mat(:));
    rows_corr = zeros(size(corr_mat,1),1);
    cols_corr = zeros(size(corr_mat,2),1);
    count = 0;
    for i=1:size(corr_mat,1)
        j = 1;
        while j < size(corr_mat,2)
            if corr_mat(i,j) > thresh_value*0.9 && corr_mat(i,j) <= thresh_value % experimental value
                count = count+1;   
                rows_corr(count) = i;
                cols_corr(count) = j;
                j = j+avgStaffDist;
            else
                j = j+1;
            end
        end
    end

    cols_corr2 = round(cols_corr./avgStaffDist);
    temp = find(cols_corr2>0);
    temp = max(temp);
    cols_corr2 = cols_corr2(1:temp);
    rows_corr = rows_corr(1:temp);
    C = unique(cols_corr2);
    rows_corr_8=[];
    cols_corr_8=[];
    for i=1:length(C)
        temp = find(cols_corr2==C(i));
        temp = temp(1);
        rows_corr_8(i) = rows_corr(temp) - round(0.5*avgStaffDist);
        cols_corr_8(i) = cols_corr(temp) - round(1.5*avgStaffDist);
    end
    SE=strel('line',3,0);
    J = imclose(J,SE);                                                     % Bridging
    imshow(J),figure
    J = ~bwareaopen(~J, 400,8);                                            % Hole filling for full notes
    imshow(J),figure


    SE = strel('disk',11,0);
    J = imerode(J,SE);
    SE = strel('disk',thickness-1,0);
    J = imdilate(J,SE);                                                    % Morphological operations to obtain note heads
    imshow(J),figure
    J_dif = bwareaopen(J,450);
    J = J-J_dif;                                                           % Removal of big objects
    imshow(J)
    LOCS2 = zeros(13 , length(LOCS)/5);                                    % Different line positions
    for k = 1:length(LOCS) / 5
        for i = 1:5
            LOCS2(i*2 + 1,k) = LOCS(i + (k-1)*5);
            LOCS2(i*2,k) = LOCS(i + (k-1)*5) - avgStaffDist/2 ;
        end
        LOCS2(1,k)  = LOCS((k-1)*5 + 1)- avgStaffDist   ;
        LOCS2(12,k) = LOCS(i + (k-1)*5) + avgStaffDist/2 ;
        LOCS2(13,k) = LOCS(i + (k-1)*5) + avgStaffDist   ;
    end
    LOCS2 = round(reshape(LOCS2,1,13*length(LOCS)/5));


    MYLOCS = zeros(13*length(LOCS)/5,40);                                  % Note positions
    for i=1:13*length(LOCS)/5
            [PKS,LOCS3] = findpeaks(J(LOCS2(i),cols_zero1+4*avgStaffDist:cols_zero2-2*avgStaffDist),'MinPeakDistance',avgStaffDist);
            for k = 1:length(LOCS3)
                MYLOCS(i,k) = LOCS3(k) + cols_zero1+4*avgStaffDist;
            end
    end


    note_indices = zeros(13*length(LOCS)/5,40,2);                          % Note indices
    for i=1:12*length(LOCS)/5
        j=1;
        while MYLOCS(i,j) > 0
            note_indices(i,j,1) = LOCS2(i);
            note_indices(i,j,2) = MYLOCS(i,j);
            j = j+1;
        end
    end


    s = [rows cols];                                                       % Finding beamed notes
    m=1;
    for j=1:13*length(LOCS)/5
        for k=1:40
            if note_indices(j,k,1) > 0 && note_indices(j,k,2) > 0
                note_positions(m) = sub2ind(s,note_indices(j,k,1),note_indices(j,k,2));
                m = m+1;
            end
        end
    end
    CC = bwconncomp(J_yedek);
    [rows2,cols2] = size(CC.PixelIdxList);
    p = 1;
    beamed_notes_temp=[];
    beamed_notes=[];
    for i=1:cols2
        temp_notecount = 0;
            for j=1:m-1
                if find(CC.PixelIdxList{1,i} == note_positions(j))>0
                    temp_notecount = temp_notecount + 1;
                    beamed_notes_temp(temp_notecount) = note_positions(j);
                end
            end
            if temp_notecount >= 2
                for k = 1:temp_notecount
                     beamed_notes(p) = beamed_notes_temp(k);
                     p = p+1;
                end
            end
    end
    [rows_beamed,cols_beamed] = ind2sub(s,beamed_notes);


    p = 1;
    q = 1;
    for i = 1:13*length(LOCS)/5                                            % Finding quarter and half notes
        for j = 1:40
            if MYLOCS(i,j) > 0
                R = LOCS2(i);
                T = MYLOCS(i,j);
                if  ~(length(find(rows_beamed==R))>0 && length(find(cols_beamed==T))>0)
                       mysum = sum(J_yedek(R-thickness:R+thickness,T));
                    if mysum == 2*thickness+1
                        rows_quarter(p) = R;
                        cols_quarter(p) = T;
                        p = p+1;
                    else
                        rows_half(q) = R;
                        cols_half(q) = T;
                        q = q+1;
                    end
                end
            end
        end
    end

    rows_eigth=[];
    cols_eigth=[];

    if ~isempty(rows_corr_8)&&~isempty(cols_corr_8)
    w = 1;
    for i=1:length(rows_quarter)
            for j=1:length(rows_corr_8)
                if (rows_quarter(i)-round(avgStaffDist/2)<rows_corr_8(j))&&(rows_quarter(i)+round(avgStaffDist/2)>rows_corr_8(j))&&(cols_quarter(i)-round(avgStaffDist/2)<cols_corr_8(j))&&(cols_quarter(i)+round(avgStaffDist/2)>cols_corr_8(j))
                    rows_eigth(w) = rows_quarter(i);
                    cols_eigth(w) = cols_quarter(i);
                    w = w+1;
                end
            end
    end
    end


    s = [13 40];                                                           % Transform all matrix to a playable form
    cnt = 1;
    for i=1:length(LOCS)/5
        myBlock = MYLOCS((i-1)*13+1:i*13,:); 
        myBlock2 = reshape(myBlock,1,520);
        myBlock2 = sort(myBlock2);
        firstnonzero = find(myBlock2>0);
        myBlock2 = myBlock2(firstnonzero(1):end);
        for j=1:length(myBlock2)
            [r, c] = ind2sub(s,find(myBlock==myBlock2(j)));
            myNotes(cnt,1) = r;
            if ~isempty(find(rows_beamed==LOCS2((i-1)*13+r)))&&~isempty(find(cols_beamed==myBlock(r,c)))
                myNotes(cnt,2) = 1;
            end
            if ~isempty(find(rows_quarter==LOCS2((i-1)*13+r)))&&~isempty(find(cols_quarter==myBlock(r,c)))
                myNotes(cnt,2) = 2;
            end
            if ~isempty(find(rows_eigth==LOCS2((i-1)*13+r)))&&~isempty(find(cols_eigth==myBlock(r,c)))
                myNotes(cnt,2) = 1;
            end
            if ~isempty(find(rows_half==LOCS2((i-1)*13+r)))&&~isempty(find(cols_half==myBlock(r,c)))
                myNotes(cnt,2) = 4;
            end
            cnt = cnt + 1;
        end
    end
end

