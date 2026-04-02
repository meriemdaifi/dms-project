%% DMS ECU Decision Logic - Stateflow Chart Definition
%% This script defines the state machine logic for the ECU
%% as it would be implemented in a Simulink/Stateflow model.
%%
%% States:
%%   - Normal: Driver is attentive
%%   - FatigueWarning: Early fatigue signs detected
%%   - FatigueCritical: Significant fatigue detected
%%   - DrowsinessEmergency: Driver is dangerously drowsy
%%   - DistractionWarning: Driver looking away from road
%%   - DistractionCritical: Sustained distraction detected
%%
%% Transitions are based on fatigue_score and distraction_score.

fprintf('=== ECU State Machine Definition ===\n\n');

%% State Definitions
states = {
    'Normal',              0, 'Driver attentive, no alert needed';
    'FatigueWarning',      2, 'Early fatigue signs, visual warning';
    'FatigueCritical',     3, 'Significant fatigue, audible alert';
    'DrowsinessEmergency', 4, 'Drowsiness detected, emergency brake assist';
    'DistractionWarning',  2, 'Distraction detected, visual warning';
    'DistractionCritical', 3, 'Sustained distraction, audible alert';
};

fprintf('States:\n');
for i = 1:size(states, 1)
    fprintf('  [%d] %s (Alert Level %d): %s\n', i, states{i,1}, states{i,2}, states{i,3});
end

%% Transition Table
fprintf('\nTransition Table:\n');
fprintf('%-25s -> %-25s : Condition\n', 'From', 'To');
fprintf('%s\n', repmat('-', 1, 80));

transitions = {
    'Normal',              'FatigueWarning',      'fatigue > 0.35';
    'Normal',              'DistractionWarning',   'distraction > 0.35';
    'FatigueWarning',      'FatigueCritical',     'fatigue > 0.6';
    'FatigueWarning',      'Normal',              'fatigue < 0.2';
    'FatigueCritical',     'DrowsinessEmergency', 'fatigue > 0.75';
    'FatigueCritical',     'FatigueWarning',      'fatigue < 0.5';
    'DrowsinessEmergency', 'FatigueCritical',     'fatigue < 0.6';
    'DistractionWarning',  'DistractionCritical', 'distraction > 0.6';
    'DistractionWarning',  'Normal',              'distraction < 0.2';
    'DistractionCritical', 'DistractionWarning',  'distraction < 0.5';
};

for i = 1:size(transitions, 1)
    fprintf('%-25s -> %-25s : %s\n', transitions{i,1}, transitions{i,2}, transitions{i,3});
end

%% ECU Actions
fprintf('\nECU Actions per State:\n');
fprintf('%-25s : %s\n', 'Normal', 'Green status LED, no alert');
fprintf('%-25s : %s\n', 'FatigueWarning', 'Yellow LED, dashboard warning icon');
fprintf('%-25s : %s\n', 'FatigueCritical', 'Red LED, audible beep, suggest break');
fprintf('%-25s : %s\n', 'DrowsinessEmergency', 'Flashing red, continuous alarm, brake assist');
fprintf('%-25s : %s\n', 'DistractionWarning', 'Yellow LED, HUD arrow indicator');
fprintf('%-25s : %s\n', 'DistractionCritical', 'Red LED, audible alert, steering vibration');

fprintf('\n=== ECU State Machine Definition Complete ===\n');
