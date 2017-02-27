function [updrsNew] = dataLoading(updrs, nOfPatients, nOfDays)
% This function fixes the original matrix downloaded from the website:
% - updrs --> original matrix
% - nOfPatients --> number of patients in the dataset
% - nOfDays --> number of days of observation
% - updrsNew --> new matrix with the requested features
% [updrsNew] = dataLoading(updrs, nOfPatients, nOfDays)

    count = 1;
    lung = 0;
    for patient = 1:nOfPatients
        %patientIndex = [];  
        timeArray = [];
        patientIndex = find(updrs(:, 1) == patient);    % Index-Patient vector

        for k = 1:length(patientIndex)
            % timeArray --> is the vector containing all the days rounded to
            % the closest INTEGER number related to the "patient"-th patient.
            timeArray(k, 1) = floor(updrs(patientIndex(k), 4)); 
        end

        % In this cycle I evaluate the mean for each patient: I scroll every
        % day from day 1 to day 180: in this way I do not consider negative
        % days.
        for days = 1:nOfDays
            %timeIndex = [];
            sumRow = zeros(1, 22); 
            timeIndex = find(timeArray == days);    % Contains the indices of 
                                                    % the days index

            if ~isempty(timeIndex)  
                for (ii = 1:length(timeIndex))
                    sumRow = sumRow + updrs(timeIndex(ii) + lung, :);
                end
                sumRow = sumRow ./ length(timeIndex);   % <-- Contains the 
                                                        % vector of the means
                updrsNew(count, :) = sumRow;    % <-- Final matrix for the 
                                                % patient "patient"-th
                count = count + 1;
            end
        end 
        % lungh contains the index of the last element related to the
        % "patient"-th patiens.
        lung = patientIndex(length(patientIndex)); 
    end
    
end