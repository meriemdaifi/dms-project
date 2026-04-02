%% DMS Simulink Model - MATLAB Script
%% This script creates and configures a Simulink model for the DMS system.
%% It defines the block diagram architecture and signal flow.
%%
%% Since Simulink requires MATLAB with the Simulink toolbox,
%% this script programmatically defines the model structure
%% that can be opened and run in MATLAB/Simulink.

%% =====================================================
%% DMS System Parameters
%% =====================================================

% Sampling parameters
Ts = 1/30;  % Sampling period (30 FPS)
simulation_time = 120;  % Total simulation time in seconds

% Fatigue detection thresholds
PERCLOS_WARNING = 40;       % PERCLOS percentage for warning
PERCLOS_CRITICAL = 70;      % PERCLOS percentage for critical alert
FATIGUE_WARNING = 0.35;     % Fatigue score warning threshold
FATIGUE_CRITICAL = 0.6;     % Fatigue score critical threshold
DROWSINESS_THRESHOLD = 0.75; % Drowsiness emergency threshold

% Distraction detection thresholds
YAW_THRESHOLD = 25;         % Maximum acceptable yaw (degrees)
PITCH_THRESHOLD = 15;       % Maximum acceptable pitch (degrees)
DISTRACTION_WARNING = 0.35; % Distraction score warning threshold
DISTRACTION_CRITICAL = 0.6; % Distraction score critical threshold

% Eye parameters
EAR_THRESHOLD = 0.25;       % Eye Aspect Ratio threshold for blink detection
BLINK_DURATION_NORMAL = 0.15; % Normal blink duration (seconds)
LONG_CLOSURE_THRESHOLD = 2.0; % Prolonged eye closure threshold (seconds)

% PERCLOS calculation window
PERCLOS_WINDOW = 60;        % Window size in seconds for PERCLOS calculation

fprintf('DMS System Parameters loaded.\n');
fprintf('Sampling rate: %.0f Hz\n', 1/Ts);
fprintf('Simulation time: %.0f seconds\n', simulation_time);

%% =====================================================
%% Create Simulink Model Programmatically
%% =====================================================

model_name = 'dms_simulink_model';

% Check if Simulink is available
if exist('new_system', 'file')
    fprintf('Creating Simulink model: %s\n', model_name);

    % Close existing model if open
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end

    % Create new model
    new_system(model_name);
    open_system(model_name);

    % Set model parameters
    set_param(model_name, 'StopTime', num2str(simulation_time));
    set_param(model_name, 'FixedStep', num2str(Ts));
    set_param(model_name, 'Solver', 'FixedStepDiscrete');

    %% ---- Video Acquisition Block ----
    add_block('simulink/Sources/From Workspace', ...
        [model_name '/Video_Input'], ...
        'Position', [50, 100, 150, 140], ...
        'VariableName', 'eye_state_signal');

    add_block('simulink/Sources/From Workspace', ...
        [model_name '/Head_Yaw_Input'], ...
        'Position', [50, 200, 150, 240], ...
        'VariableName', 'yaw_signal');

    add_block('simulink/Sources/From Workspace', ...
        [model_name '/Head_Pitch_Input'], ...
        'Position', [50, 300, 150, 340], ...
        'VariableName', 'pitch_signal');

    %% ---- PERCLOS Calculation Block ----
    add_block('simulink/Discrete/Discrete Filter', ...
        [model_name '/PERCLOS_Filter'], ...
        'Position', [250, 100, 350, 140], ...
        'Numerator', mat2str(ones(1, round(PERCLOS_WINDOW/Ts)) / round(PERCLOS_WINDOW/Ts)), ...
        'Denominator', '1');

    %% ---- Fatigue Score Calculator (Subsystem) ----
    add_block('simulink/Commonly Used Blocks/Subsystem', ...
        [model_name '/Fatigue_Calculator'], ...
        'Position', [450, 80, 600, 180]);

    %% ---- Distraction Score Calculator ----
    add_block('simulink/Commonly Used Blocks/Subsystem', ...
        [model_name '/Distraction_Calculator'], ...
        'Position', [450, 220, 600, 320]);

    %% ---- ECU Decision Logic ----
    add_block('simulink/Commonly Used Blocks/Subsystem', ...
        [model_name '/ECU_Decision'], ...
        'Position', [700, 150, 850, 250]);

    %% ---- Alert Output ----
    add_block('simulink/Sinks/To Workspace', ...
        [model_name '/Alert_Output'], ...
        'Position', [950, 180, 1050, 220], ...
        'VariableName', 'alert_level', ...
        'SaveFormat', 'Array');

    add_block('simulink/Sinks/Scope', ...
        [model_name '/DMS_Scope'], ...
        'Position', [950, 100, 1000, 140]);

    %% ---- Connect blocks ----
    add_line(model_name, 'Video_Input/1', 'PERCLOS_Filter/1');
    add_line(model_name, 'PERCLOS_Filter/1', 'Fatigue_Calculator/1');
    add_line(model_name, 'Head_Yaw_Input/1', 'Distraction_Calculator/1');
    add_line(model_name, 'Fatigue_Calculator/1', 'ECU_Decision/1');
    add_line(model_name, 'Distraction_Calculator/1', 'ECU_Decision/2');
    add_line(model_name, 'ECU_Decision/1', 'Alert_Output/1');
    add_line(model_name, 'ECU_Decision/1', 'DMS_Scope/1');

    % Save model
    save_system(model_name);
    fprintf('Simulink model saved: %s.slx\n', model_name);
else
    fprintf('Simulink not available. Model structure defined in script.\n');
    fprintf('To use this model, open it in MATLAB with Simulink toolbox.\n');
end

%% =====================================================
%% Model Architecture Description (for documentation)
%% =====================================================
fprintf('\n=== DMS Simulink Model Architecture ===\n');
fprintf('Block 1: Video Input (From Workspace) -> Eye state signal\n');
fprintf('Block 2: Head Yaw Input (From Workspace) -> Yaw angle signal\n');
fprintf('Block 3: Head Pitch Input (From Workspace) -> Pitch angle signal\n');
fprintf('Block 4: PERCLOS Filter (Moving Average) -> PERCLOS percentage\n');
fprintf('Block 5: Fatigue Calculator (Subsystem) -> Fatigue score\n');
fprintf('Block 6: Distraction Calculator (Subsystem) -> Distraction score\n');
fprintf('Block 7: ECU Decision Logic (Subsystem) -> Alert level\n');
fprintf('Block 8: Alert Output (To Workspace) -> Alert logging\n');
fprintf('Block 9: DMS Scope -> Real-time visualization\n');
