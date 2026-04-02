%% DMS Simulation Script
%% Runs the Driver Monitoring System simulation in MATLAB
%% Generates test signals and processes them through the DMS pipeline
%%
%% This script can run standalone (without Simulink) to validate
%% the DMS algorithms in MATLAB.

clear; clc; close all;

fprintf('=== DMS MATLAB Simulation ===\n\n');

%% =====================================================
%% System Parameters
%% =====================================================
Fs = 30;                    % Sampling frequency (Hz)
Ts = 1/Fs;                  % Sampling period
T_sim = 120;                % Simulation duration (seconds)
t = 0:Ts:T_sim-Ts;          % Time vector
N = length(t);              % Number of samples

% Thresholds
PERCLOS_WINDOW = 60 * Fs;   % PERCLOS window (samples)
FATIGUE_WARNING = 0.35;
FATIGUE_CRITICAL = 0.6;
DROWSY_THRESHOLD = 0.75;
DISTRACTION_WARNING = 0.35;
DISTRACTION_CRITICAL = 0.6;
YAW_THRESHOLD = 25;         % degrees

fprintf('Fs = %d Hz, Duration = %d s, Samples = %d\n', Fs, T_sim, N);

%% =====================================================
%% Scenario 1: Attentive Driver
%% =====================================================
fprintf('\n--- Scenario 1: Attentive Driver ---\n');

% Generate eye state (1 = open, 0 = closed)
eye_state_1 = ones(1, N);
blink_interval = round(3 * Fs);  % Blink every 3 seconds
blink_duration = round(0.15 * Fs);  % 150ms blink
for i = 1:blink_interval:N
    end_idx = min(i + blink_duration - 1, N);
    eye_state_1(i:end_idx) = 0;
end

% Head pose (small natural movements)
yaw_1 = 3 * sin(0.1 * t) + randn(1, N) * 1.5;
yaw_1 = max(min(yaw_1, 15), -15);
pitch_1 = 2 * sin(0.15 * t) + randn(1, N) * 1.0;
pitch_1 = max(min(pitch_1, 10), -10);

%% =====================================================
%% Scenario 2: Fatigued Driver
%% =====================================================
fprintf('--- Scenario 2: Fatigued Driver ---\n');

eye_state_2 = ones(1, N);
for i = 1:N
    progress = i / N;
    if progress < 0.25
        % Phase 1: Normal
        if mod(i, round(3*Fs)) < round(0.2*Fs)
            eye_state_2(i) = 0;
        end
    elseif progress < 0.5
        % Phase 2: Longer blinks
        if mod(i, round(2.5*Fs)) < round(0.4*Fs)
            eye_state_2(i) = 0;
        end
        if mod(i, round(15*Fs)) < round(1.5*Fs)
            eye_state_2(i) = 0;
        end
    elseif progress < 0.75
        % Phase 3: Frequent closures
        if mod(i, round(2*Fs)) < round(0.6*Fs)
            eye_state_2(i) = 0;
        end
        if mod(i, round(10*Fs)) < round(2*Fs)
            eye_state_2(i) = 0;
        end
    else
        % Phase 4: Extended closures
        if mod(i, round(1.5*Fs)) < round(0.8*Fs)
            eye_state_2(i) = 0;
        end
        if mod(i, round(8*Fs)) < round(3*Fs)
            eye_state_2(i) = 0;
        end
    end
end

yaw_2 = 2 * sin(0.05 * t) + randn(1, N) .* (2 + 3 * (t/T_sim));
pitch_2 = -5 * (t/T_sim) + randn(1, N) .* (2 + 2 * (t/T_sim));

%% =====================================================
%% Scenario 3: Distracted Driver
%% =====================================================
fprintf('--- Scenario 3: Distracted Driver ---\n');

eye_state_3 = ones(1, N);
yaw_3 = zeros(1, N);
pitch_3 = zeros(1, N);

for i = 1:N
    progress = i / N;
    % Normal blinking
    if mod(i, round(4*Fs)) < round(0.15*Fs)
        eye_state_3(i) = 0;
    end

    if progress < 0.25
        yaw_3(i) = 3 * sin(0.1 * t(i)) + randn * 2;
        pitch_3(i) = 2 * sin(0.15 * t(i)) + randn * 1.5;
    elseif progress < 0.5
        if mod(i, round(10*Fs)) < round(3*Fs)
            yaw_3(i) = 40 + randn * 5;
            pitch_3(i) = -10 + randn * 3;
        else
            yaw_3(i) = randn * 3;
            pitch_3(i) = randn * 2;
        end
    elseif progress < 0.75
        if mod(i, round(7*Fs)) < round(4*Fs)
            yaw_3(i) = 50 * (-1)^floor(i/(7*Fs)) + randn * 5;
            pitch_3(i) = -15 + randn * 3;
        else
            yaw_3(i) = randn * 5;
            pitch_3(i) = randn * 3;
        end
    else
        yaw_3(i) = 30 * sin(0.3 * t(i)) + randn * 8;
        pitch_3(i) = -25 + randn * 5;
    end
end

yaw_3 = max(min(yaw_3, 90), -90);
pitch_3 = max(min(pitch_3, 45), -45);

%% =====================================================
%% PERCLOS Calculation
%% =====================================================
fprintf('\nCalculating PERCLOS...\n');

perclos_1 = compute_perclos(eye_state_1, PERCLOS_WINDOW);
perclos_2 = compute_perclos(eye_state_2, PERCLOS_WINDOW);
perclos_3 = compute_perclos(eye_state_3, PERCLOS_WINDOW);

%% =====================================================
%% Fatigue Score Calculation
%% =====================================================
fprintf('Calculating fatigue scores...\n');

fatigue_1 = compute_fatigue_score(perclos_1);
fatigue_2 = compute_fatigue_score(perclos_2);
fatigue_3 = compute_fatigue_score(perclos_3);

%% =====================================================
%% Distraction Score Calculation
%% =====================================================
fprintf('Calculating distraction scores...\n');

distraction_1 = compute_distraction_score(yaw_1, pitch_1, YAW_THRESHOLD);
distraction_2 = compute_distraction_score(yaw_2, pitch_2, YAW_THRESHOLD);
distraction_3 = compute_distraction_score(yaw_3, pitch_3, YAW_THRESHOLD);

%% =====================================================
%% ECU Alert Level Calculation
%% =====================================================
fprintf('Computing ECU alert levels...\n');

alert_1 = compute_alert_level(fatigue_1, distraction_1, FATIGUE_WARNING, FATIGUE_CRITICAL, DROWSY_THRESHOLD, DISTRACTION_WARNING, DISTRACTION_CRITICAL);
alert_2 = compute_alert_level(fatigue_2, distraction_2, FATIGUE_WARNING, FATIGUE_CRITICAL, DROWSY_THRESHOLD, DISTRACTION_WARNING, DISTRACTION_CRITICAL);
alert_3 = compute_alert_level(fatigue_3, distraction_3, FATIGUE_WARNING, FATIGUE_CRITICAL, DROWSY_THRESHOLD, DISTRACTION_WARNING, DISTRACTION_CRITICAL);

%% =====================================================
%% Visualization
%% =====================================================
fprintf('\nGenerating plots...\n');

% Figure 1: Attentive Driver
figure('Name', 'Scenario 1: Attentive Driver', 'Position', [50, 50, 1200, 800]);
plot_scenario(t, eye_state_1, yaw_1, perclos_1, fatigue_1, distraction_1, alert_1, 'Attentive Driver');
saveas(gcf, 'attentive_scenario_matlab.png');

% Figure 2: Fatigued Driver
figure('Name', 'Scenario 2: Fatigued Driver', 'Position', [100, 50, 1200, 800]);
plot_scenario(t, eye_state_2, yaw_2, perclos_2, fatigue_2, distraction_2, alert_2, 'Fatigued Driver');
saveas(gcf, 'fatigued_scenario_matlab.png');

% Figure 3: Distracted Driver
figure('Name', 'Scenario 3: Distracted Driver', 'Position', [150, 50, 1200, 800]);
plot_scenario(t, eye_state_3, yaw_3, perclos_3, fatigue_3, distraction_3, alert_3, 'Distracted Driver');
saveas(gcf, 'distracted_scenario_matlab.png');

% Figure 4: Comparison Dashboard
figure('Name', 'DMS Comparison Dashboard', 'Position', [200, 50, 1400, 900]);
subplot(3, 3, 1); plot(t, fatigue_1, 'b'); title('Fatigue - Attentive'); ylabel('Score'); ylim([0 1]); grid on;
subplot(3, 3, 2); plot(t, fatigue_2, 'r'); title('Fatigue - Fatigued'); ylim([0 1]); grid on;
subplot(3, 3, 3); plot(t, fatigue_3, 'm'); title('Fatigue - Distracted'); ylim([0 1]); grid on;
subplot(3, 3, 4); plot(t, distraction_1, 'b'); title('Distraction - Attentive'); ylabel('Score'); ylim([0 1]); grid on;
subplot(3, 3, 5); plot(t, distraction_2, 'r'); title('Distraction - Fatigued'); ylim([0 1]); grid on;
subplot(3, 3, 6); plot(t, distraction_3, 'm'); title('Distraction - Distracted'); ylim([0 1]); grid on;
subplot(3, 3, 7); plot(t, alert_1, 'b'); title('Alert - Attentive'); ylabel('Level'); xlabel('Time (s)'); ylim([0 4]); grid on;
subplot(3, 3, 8); plot(t, alert_2, 'r'); title('Alert - Fatigued'); xlabel('Time (s)'); ylim([0 4]); grid on;
subplot(3, 3, 9); plot(t, alert_3, 'm'); title('Alert - Distracted'); xlabel('Time (s)'); ylim([0 4]); grid on;
sgtitle('DMS Comparison Dashboard');
saveas(gcf, 'comparison_dashboard_matlab.png');

fprintf('\n=== Simulation Complete ===\n');
fprintf('Plots saved as PNG files.\n');

%% =====================================================
%% Helper Functions
%% =====================================================

function perclos = compute_perclos(eye_state, window)
    % Compute PERCLOS (percentage of eye closure) using moving average
    N = length(eye_state);
    perclos = zeros(1, N);
    eye_closed = 1 - eye_state;  % 1 when closed

    for i = 1:N
        start_idx = max(1, i - window + 1);
        perclos(i) = mean(eye_closed(start_idx:i)) * 100;
    end
end

function fatigue = compute_fatigue_score(perclos)
    % Compute fatigue score from PERCLOS (normalized 0-1)
    fatigue = min(perclos / 100, 1);
end

function distraction = compute_distraction_score(yaw, pitch, yaw_thresh)
    % Compute distraction score from head pose
    N = length(yaw);
    distraction = zeros(1, N);
    window = 60;  % 2-second window at 30Hz

    for i = 1:N
        start_idx = max(1, i - window + 1);
        yaw_component = min(mean(abs(yaw(start_idx:i))) / 90, 1);
        pitch_component = min(mean(abs(pitch(start_idx:i))) / 90, 1);
        looking_away = mean(abs(yaw(start_idx:i)) > yaw_thresh);
        distraction(i) = 0.4 * yaw_component + 0.2 * pitch_component + 0.4 * looking_away;
    end
end

function alert = compute_alert_level(fatigue, distraction, fw, fc, dt, dw, dc)
    % Compute ECU alert level
    % 0=None, 1=Info, 2=Warning, 3=Critical, 4=Emergency
    N = length(fatigue);
    alert = zeros(1, N);

    for i = 1:N
        if fatigue(i) > dt
            alert(i) = 4;  % Emergency
        elseif fatigue(i) > fc
            alert(i) = 3;  % Critical
        elseif distraction(i) > dc
            alert(i) = 3;  % Critical
        elseif fatigue(i) > fw
            alert(i) = 2;  % Warning
        elseif distraction(i) > dw
            alert(i) = 2;  % Warning
        else
            alert(i) = 0;  % None
        end
    end
end

function plot_scenario(t, eye_state, yaw, perclos, fatigue, distraction, alert, scenario_name)
    % Plot all metrics for a scenario
    subplot(5, 1, 1);
    area(t, eye_state, 'FaceColor', [0.4 0.8 0.4], 'EdgeColor', 'none');
    title([scenario_name ' - Eye State']); ylabel('State');
    yticks([0 1]); yticklabels({'Closed', 'Open'}); grid on;

    subplot(5, 1, 2);
    plot(t, yaw, 'b', 'LineWidth', 0.5);
    hold on;
    yline(25, 'r--', 'Threshold');
    yline(-25, 'r--');
    hold off;
    title('Head Yaw'); ylabel('Degrees'); grid on;

    subplot(5, 1, 3);
    plot(t, perclos, 'Color', [1 0.6 0], 'LineWidth', 1.5);
    hold on;
    yline(40, 'r--', 'Warning');
    yline(70, 'r--', 'Critical');
    hold off;
    title('PERCLOS'); ylabel('%'); grid on;

    subplot(5, 1, 4);
    plot(t, fatigue, 'r', 'LineWidth', 1.5);
    hold on;
    plot(t, distraction, 'b', 'LineWidth', 1.5);
    yline(0.35, 'k--', 'Warning');
    yline(0.6, 'r--', 'Critical');
    hold off;
    title('Fatigue & Distraction Scores');
    ylabel('Score'); legend('Fatigue', 'Distraction'); ylim([0 1]); grid on;

    subplot(5, 1, 5);
    stairs(t, alert, 'k', 'LineWidth', 1.5);
    title('ECU Alert Level');
    ylabel('Level'); xlabel('Time (s)');
    yticks([0 1 2 3 4]);
    yticklabels({'None', 'Info', 'Warning', 'Critical', 'Emergency'});
    ylim([-0.5 4.5]); grid on;

    sgtitle(scenario_name, 'FontSize', 14, 'FontWeight', 'bold');
end
