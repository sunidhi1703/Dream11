import pandas as pd
import numpy as np
import json
import os
import re
import xgboost as xgb
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
from collections import defaultdict

# --- Setup ---
warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app) 

# --- Global Variables (Loaded once on startup to be fast) ---
df_logs = None
df_roles_season = None
df_roles_global = None
model = None
FEATURES = []
role_median_credits = {}

# ==================================================================
# --- ALL HELPER FUNCTIONS ---
# ==================================================================

def get_player_role(player_id, season):
    try:
        seasonal_role = df_roles_season[
            (df_roles_season['player_id'] == str(player_id)) & 
            (df_roles_season['season'] == int(season))
        ]
        if not seasonal_role.empty:
            return seasonal_role.iloc[0]['role']
        
        global_role = df_roles_global[df_roles_global['player_id'] == str(player_id)]
        if not global_role.empty:
            return global_role.iloc[0]['role']
    except Exception as e:
        print(f"Warning: Error in get_player_role for {player_id}, season {season}: {e}")
    return 'BAT'

def calculate_batting_points(stats):
    points = 0
    runs = stats.get('runs', 0); balls_faced = stats.get('balls_faced', 0)
    points += runs + stats.get('fours', 0) * 1 + stats.get('sixes', 0) * 2
    if runs == 0 and stats.get('is_out', False) and stats.get('role', 'BAT') in ['WK', 'BAT', 'AR']:
        points -= 2
    if balls_faced >= 10:
        sr = (runs / balls_faced) * 100 if balls_faced > 0 else 0
        if sr >= 170: points += 6
        elif 150 <= sr < 170: points += 4
        elif 130 <= sr < 150: points += 2
        elif 50 <= sr < 70: points -= 2
        elif sr < 50: points -= 4
    return points

def calculate_bowling_points(stats):
    points = 0
    wickets = stats.get('wickets', 0); overs = stats.get('overs', 0)
    points += wickets * 25 + stats.get('lbw_bowled', 0) * 8
    if wickets >= 5: points += 16
    elif wickets == 4: points += 10
    elif wickets == 3: points += 6
    points += stats.get('maidens', 0) * 12
    if overs >= 2:
        er = stats.get('runs_conceded', 0) / overs if overs > 0 else 0
        if er <= 5.0: points += 6
        elif 5.01 <= er <= 6.5: points += 4
        elif 6.51 <= er <= 8.0: points += 2
        elif 10.0 <= er <= 11.0: points -= 2
        elif er > 11.0: points -= 4
    return points

def calculate_fielding_points(stats):
    catches = stats.get('catches', 0)
    points = catches * 8
    if catches >= 3: points += 4
    points += stats.get('stumpings', 0) * 12 + stats.get('run_outs_direct', 0) * 12 + stats.get('run_outs_shared', 0) * 6
    return points

def get_total_fantasy_points(player_match_stats):
    return calculate_batting_points(player_match_stats) + calculate_bowling_points(player_match_stats) + calculate_fielding_points(player_match_stats)

def parse_match_data_for_dream_xi(data):
    # This is the full, robust parser from our Colab notebook
    season = int(data['info']['dates'][0].split('-')[0])
    name_to_id = {name: str(pid) for name, pid in data['info']['registry']['people'].items()}
    id_to_team = {str(pid): team for team, players in data['info']['players'].items() for name in players if (pid := name_to_id.get(name))}
    
    stats = defaultdict(lambda: defaultdict(int))
    out_status = defaultdict(bool)
    
    for inning in data['innings']:
        for over in inning['overs']:
            runs_in_this_over = 0
            bowler_of_over = None
            for ball in over['deliveries']:
                batter_id, bowler_id = name_to_id.get(ball['batter']), name_to_id.get(ball['bowler'])
                if not (batter_id and bowler_id): continue
                
                bowler_of_over = bowler_id
                runs = ball['runs']['batter']
                stats[batter_id]['runs'] += runs
                if runs == 4: stats[batter_id]['fours'] += 1
                if runs == 6: stats[batter_id]['sixes'] += 1
                
                extras = ball.get('extras', {})
                if 'wides' not in extras:
                    stats[batter_id]['balls_faced'] += 1
                    stats[bowler_id]['legal_balls_bowled'] += 1
                
                stats[bowler_id]['runs_conceded'] += runs + extras.get('wides', 0) + extras.get('noballs', 0)
                runs_in_this_over += ball['runs']['total']
                
                if 'wickets' in ball:
                    for wicket in ball['wickets']:
                        if player_out_name := wicket.get('player_out'):
                            if player_out_id := name_to_id.get(player_out_name):
                                out_status[player_out_id] = True
                        
                        kind, fielders = wicket.get('kind', ''), wicket.get('fielders', [])
                        if kind != 'run out':
                            stats[bowler_id]['wickets'] += 1
                            if kind in ['bowled', 'lbw']: stats[bowler_id]['lbw_bowled'] += 1
                            if kind == 'caught' and fielders and 'name' in fielders[0]:
                                if f_id := name_to_id.get(fielders[0]['name']): stats[f_id]['catches'] += 1
                            if kind == 'stumped' and fielders and 'name' in fielders[0]:
                                if f_id := name_to_id.get(fielders[0]['name']): stats[f_id]['stumpings'] += 1
                        else:
                             valid_fielders = [f['name'] for f in fielders if 'name' in f]
                             if len(valid_fielders) == 1:
                                 if f_id := name_to_id.get(valid_fielders[0]): stats[f_id]['run_outs_direct'] += 1
                             else:
                                 for f_name in valid_fielders:
                                     if f_id := name_to_id.get(f_name): stats[f_id]['run_outs_shared'] += 1
            
            if bowler_of_over and runs_in_this_over == 0 and stats[bowler_of_over].get('legal_balls_bowled', 0) % 6 == 0 and stats[bowler_of_over].get('legal_balls_bowled', 0) > 0:
                 stats[bowler_of_over]['maidens'] += 1

    final_stats_list = []
    for pid in id_to_team:
        player_stats = stats[pid]
        player_stats['overs'] = player_stats.pop('legal_balls_bowled', 0) / 6.0
        player_stats['is_out'] = out_status[pid]
        player_stats['role'] = get_player_role(pid, season)
        # Clean up temporary keys used for maiden calculation
        for over_num in range(20): player_stats.pop(over_num, None)
        total_fp = get_total_fantasy_points(player_stats)
        final_stats_list.append({'player_id': pid, 'fantasy_points': total_fp})
    
    print(f"DEBUG: Calculated FP for {len(final_stats_list)} players")
    if final_stats_list:
        print(f"DEBUG: Sample FP - {final_stats_list[0]}")
        print(f"DEBUG: Total FP calculated: {sum(p['fantasy_points'] for p in final_stats_list)}")
    
    return final_stats_list


def select_optimal_team(players_df, budget=100.0, point_col='predicted_fp'):
    if players_df.empty or not all(c in players_df.columns for c in ['player_id', point_col, 'credits', 'role', 'team']): return None
    player_ids = players_df['player_id'].astype(str).tolist()
    points = pd.Series(players_df[point_col].values, index=player_ids).to_dict()
    credits = pd.Series(players_df['credits'].values, index=player_ids).to_dict()
    roles = pd.Series(players_df['role'].values, index=player_ids).to_dict()
    teams_dict = pd.Series(players_df['team'].values, index=player_ids).to_dict()
    team_names = players_df['team'].unique()

    prob = LpProblem("Dream11", LpMaximize)
    player_vars = LpVariable.dicts("Player", player_ids, 0, 1, 'Binary')
    prob += lpSum(points[p] * player_vars[p] for p in player_ids)
    prob += lpSum(player_vars[p] for p in player_ids) == 11
    prob += lpSum(credits[p] * player_vars[p] for p in player_ids) <= budget
    for role, min_val, max_val in [('WK', 1, 4), ('BAT', 3, 6), ('AR', 1, 4), ('BOWL', 3, 6)]:
        prob += lpSum(player_vars[p] for p in player_ids if roles.get(p) == role) >= min_val
        prob += lpSum(player_vars[p] for p in player_ids if roles.get(p) == role) <= max_val
    for team in team_names:
        prob += lpSum(player_vars[p] for p in player_ids if teams_dict.get(p) == team) <= 7
        if len(team_names) >= 2: prob += lpSum(player_vars[p] for p in player_ids if teams_dict.get(p) == team) >= 1
            
    prob.solve()
    if LpStatus[prob.status] == 'Optimal':
        return players_df[players_df['player_id'].astype(str).isin([p for p in player_ids if player_vars[p].varValue == 1])].copy()
    return None

# ==================================================================
# --- FLASK ENDPOINT ---
# ==================================================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or not (df_logs is not None and model is not None):
        return jsonify({'error': 'File missing or server not ready'}), 400

    try:
        test_data = json.load(request.files['file'])
        test_match_date = pd.to_datetime(test_data['info']['dates'][0])
        test_season = test_match_date.year
        teams = test_data['info']['teams']
        venue = test_data['info'].get('venue', 'Unknown Venue')
        team_1_name, team_2_name = (teams[0], teams[1]) if len(teams) > 1 else ("TeamA", "TeamB")
        name_to_id = {name: str(pid) for name, pid in test_data['info']['registry']['people'].items()}
        id_to_name = {str(pid): name for name, pid in test_data['info']['registry']['people'].items()}
        all_player_ids = [name_to_id[p_name] for team in teams for p_name in test_data['info']['players'].get(team, []) if p_name in name_to_id]
        id_to_team = {pid: team for team in teams for p_name in test_data['info']['players'].get(team, []) if (pid := name_to_id.get(p_name))}
        
        test_players_features = []
        for pid_str in all_player_ids:
            player_history = df_logs[(df_logs['player_id'] == pid_str) & (df_logs['match_date'] < test_match_date)].sort_values(by='match_date', ascending=False)
            role = get_player_role(pid_str, test_season)
            opponent = team_2_name if id_to_team.get(pid_str) == team_1_name else team_1_name
            n_apps = len(player_history)
            
            if n_apps == 0:
                features = {k: 0 for k in ['fp_last_1', 'fp_rolling_5', 'fp_rolling_10', 'fp_std_10', 'fp_ewm_5', 'runs_rolling_5', 'balls_faced_rolling_5', 'wickets_rolling_5', 'overs_rolling_5', 'runs_conceded_rolling_5', 'fp_vs_opponent_mean', 'fp_vs_venue_mean']}
            else:
                features = {
                    'fp_last_1': player_history.iloc[0]['fantasy_points'], 'fp_rolling_5': player_history.head(5)['fantasy_points'].mean(),
                    'fp_rolling_10': player_history.head(10)['fantasy_points'].mean(), 'fp_std_10': player_history.head(10)['fantasy_points'].std(),
                    'fp_ewm_5': player_history['fantasy_points'].iloc[::-1].ewm(span=5, adjust=False).mean().iloc[-1], 'runs_rolling_5': player_history.head(5)['runs'].mean(),
                    'balls_faced_rolling_5': player_history.head(5)['balls_faced'].mean(), 'wickets_rolling_5': player_history.head(5)['wickets'].mean(),
                    'overs_rolling_5': player_history.head(5)['overs'].mean(), 'runs_conceded_rolling_5': player_history.head(5)['runs_conceded'].mean()
                }
                history_vs_opponent = player_history[player_history['opponent'] == opponent]
                features['fp_vs_opponent_mean'] = history_vs_opponent['fantasy_points'].mean() if not history_vs_opponent.empty else features['fp_rolling_5']
                history_vs_venue = player_history[player_history['venue'] == venue]
                features['fp_vs_venue_mean'] = history_vs_venue['fantasy_points'].mean() if not history_vs_venue.empty else features['fp_rolling_5']
            
            features = {k: 0 if pd.isna(v) else v for k, v in features.items()}
            median = role_median_credits.get(role, 8.0)
            credits = np.clip(median, median - 0.5, median + 0.5) if n_apps < 10 else round(np.clip(np.interp(0.7 * features.get('fp_rolling_10',0) + 0.3 * (features.get('fp_rolling_10',0) - features.get('fp_std_10',0)), [10, 60], [7.0, 10.5]), 4.0, 11.0), 2)
            
            test_players_features.append({'player_id': pid_str, 'player_name': id_to_name.get(pid_str, 'Unknown'), 'team': id_to_team.get(pid_str), 'role': role, 'opponent': opponent, 'venue': venue, 'credits': credits, **features})
        
        if not test_players_features: return jsonify({'error': 'No player data could be processed.'}), 400
        df_test = pd.DataFrame(test_players_features)
        df_test_encoded = pd.get_dummies(df_test, columns=['role', 'opponent', 'venue'])
        
        X_test_aligned = pd.DataFrame(columns=FEATURES)
        X_test_aligned = pd.concat([X_test_aligned, df_test_encoded[df_test_encoded.columns.intersection(FEATURES)]], ignore_index=True)
        for col in FEATURES:
             if col not in X_test_aligned.columns:
                 X_test_aligned[col] = 0
        X_test_aligned = X_test_aligned[FEATURES].fillna(0)
        
        df_test['predicted_fp'] = model.predict(X_test_aligned)
        
        recommended_xi = select_optimal_team(df_test, point_col='predicted_fp')
        actual_stats = parse_match_data_for_dream_xi(test_data)
        if not actual_stats: return jsonify({'error': 'Could not parse stats to determine Dream XI.'}), 500
             
        df_actual_points = pd.DataFrame(actual_stats).rename(columns={'fantasy_points': 'actual_fp'})
        df_actual_points['player_id'] = df_actual_points['player_id'].astype(str)
        df_ground_truth = df_test.merge(df_actual_points, on='player_id', how='left')
        
        # FIX: Ensure actual_fp column exists and handle missing values
        if 'actual_fp' not in df_ground_truth.columns:
            df_ground_truth['actual_fp'] = 0
        df_ground_truth['actual_fp'] = df_ground_truth['actual_fp'].fillna(0)
        
        if df_ground_truth.empty: 
            return jsonify({'error': 'Ground truth data is empty.'}), 500
             
        dream_xi = select_optimal_team(df_ground_truth, point_col='actual_fp')
        
        if recommended_xi is not None and dream_xi is not None:
            print(f"\n🔍 DEBUG: Predicted XI has {len(recommended_xi)} players")
            print(f"🔍 DEBUG: Dream XI has {len(dream_xi)} players")
            print(f"🔍 DEBUG: Actual stats parsed for {len(df_actual_points)} players")
            print(f"🔍 DEBUG: Sample actual points: {df_actual_points.head(3).to_dict('records')}")
            
            # FIX: Merge actual points for predicted XI
            pred_xi_with_actual = recommended_xi.merge(df_actual_points, on='player_id', how='left')
            if 'actual_fp' not in pred_xi_with_actual.columns:
                print("⚠️ WARNING: 'actual_fp' column missing after merge for predicted XI")
                pred_xi_with_actual['actual_fp'] = 0
            pred_xi_with_actual['actual_fp'] = pred_xi_with_actual['actual_fp'].fillna(0)
            pred_xi_actual_points = pred_xi_with_actual['actual_fp'].sum()
            
            print(f"✅ Predicted XI actual points sum: {pred_xi_actual_points}")
            
            # FIX: Dream XI already has actual_fp from ground_truth - DON'T merge again!
            # Just use the existing actual_fp column
            dream_xi_with_actual_points = dream_xi.copy()
            if 'actual_fp' not in dream_xi_with_actual_points.columns:
                print("⚠️ WARNING: 'actual_fp' column missing in dream XI")
                dream_xi_with_actual_points['actual_fp'] = 0
            dream_xi_with_actual_points['actual_fp'] = dream_xi_with_actual_points['actual_fp'].fillna(0)
            dream_xi_total_points = dream_xi_with_actual_points['actual_fp'].sum()
            
            print(f"✅ Dream XI actual points sum: {dream_xi_total_points}")

            final_score = abs(dream_xi_total_points - pred_xi_actual_points)
            
            print(f"📊 Final Score (Absolute Error): {final_score}")
            
            # Convert DataFrames to standard Python dicts before sending
            response_data = {
                'predictedXI': json.loads(pred_xi_with_actual.to_json(orient='records')),
                'dreamXI': json.loads(dream_xi_with_actual_points.to_json(orient='records')),
                'finalScore': float(final_score) 
            }
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Solver failed to generate a valid team.'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

def load_all_data():
    global df_logs, df_roles_season, df_roles_global, model, FEATURES, role_median_credits
    print("Loading all data assets...")
    try:
        # 1. Load base data files
        df_logs = pd.read_csv('player_match_logs.csv', dtype={'player_id': str})
        df_logs['match_date'] = pd.to_datetime(df_logs['match_date'])
        df_roles_season = pd.read_csv('player_roles_by_season.csv', dtype={'player_id': str, 'season': str})
        df_roles_season['season'] = pd.to_numeric(df_roles_season['season'], errors='coerce').dropna().astype(int)
        df_roles_global = pd.read_csv('player_roles_global.csv', dtype={'player_id': str})
        
        # 2. Add 'opponent' and 'venue' to df_logs
        if 'venue' not in df_logs.columns: df_logs['venue'] = 'Unknown Venue'
        teams_per_match = df_logs.groupby('match_id')['team'].unique().to_frame()
        teams_per_match['team_1'] = teams_per_match['team'].apply(lambda x: x[0] if len(x) > 0 else None)
        teams_per_match['team_2'] = teams_per_match['team'].apply(lambda x: x[1] if len(x) > 1 else None)
        df_logs = df_logs.merge(teams_per_match.drop(columns=['team']), on='match_id', how='left')
        df_logs['opponent'] = np.where(df_logs['team'] == df_logs['team_1'], df_logs['team_2'], df_logs['team_1'])
        df_logs.dropna(subset=['opponent'], inplace=True)

        # 3. Load model
        model = xgb.XGBRegressor()
        model.load_model('model_artifacts/ProductUI_Model.json')
        
        # 4. Load the final training data to get FEATURES list and calculate medians
        df_train_data = pd.read_csv('model_training_data_with_credits.csv', dtype={'player_id': str})
        
        TARGET = 'fantasy_points'
        COLS_TO_EXCLUDE = [TARGET, 'match_id', 'player_id', 'match_date', 'role', 'appearance_count', 'composite_score', 'percentile_rank']
        FEATURES = [col for col in df_train_data.columns if col not in COLS_TO_EXCLUDE]
        
        # FIX: Calculate role median credits with proper error handling
        required_median_cols = ['appearance_count', 'role', 'credits']
        if all(col in df_train_data.columns for col in required_median_cols):
            df_for_medians = df_train_data.dropna(subset=required_median_cols)
            temp_credits_df = df_for_medians[df_for_medians['appearance_count'] >= 10]
            role_median_credits = temp_credits_df.groupby('role')['credits'].median().to_dict()
        else:
             print("Warning: 'appearance_count' or 'role' not in training CSV. Using default medians.")
             role_median_credits = {'WK': 8.0, 'BAT': 8.0, 'AR': 8.5, 'BOWL': 8.0}

        print("All assets loaded successfully!")
        print(f"Loaded {len(FEATURES)} features.")
        print(f"Calculated role medians: {role_median_credits}")

    except FileNotFoundError as e:
        print(f"🚨 FATAL ERROR loading data: {e}. Make sure all required CSV and model files are present.")
        exit()
    except Exception as e:
        print(f"🚨 FATAL ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        exit()

if __name__ == '__main__':
    load_all_data()
    app.run(debug=True, port=5001)